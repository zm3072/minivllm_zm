import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

# 简单模型
# 它是一个 KV cache 的分页管理器：
# 把每条序列的 token 切成逻辑块，再把逻辑块映射到有限数量的物理块；
# 能复用已有前缀缓存就复用，不能复用就分配新块，序列结束后再释放。

# BlockManager 负责管理:
        # 谁用了哪些 KV cache block；
        # 哪些 block 空闲；
        # 哪些 block 可以根据前缀复用；
        # 哪些 block 可以安全释放。


# 物理块
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        # 这个 block 对应 token 内容的 hash，用于 prefix cache 查找
        self.hash = -1 

        # 引用计数 有多少条序列正在使用这个 Block
        self.ref_count = 0
        
        # 这个 block 里存的 token_ids
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    # 重置 block 信息
    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []




class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block
        self.block_size: int = block_size

        # list of all blocks
        # 创建并保存 所有物理块
        # [Block(0), Block(1), Block(2), ...]
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        
        # hash to block id: this is for prefix caching
        # hash -> block_id
        self.hash_to_block_id: dict[int, int] = {}

        # free block ids
        # 每次都取队头的 block
        # 用完放回队尾
        # 因此用 deque
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        # 只需要快速判断 当前块是否在使用中 用 set 更快
        self.used_block_ids: set[int] = set()

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        # 这里的 hash 的计算考虑是
        # 上下文不同, 位置之前的内容不同
        # attention 看到的历史不同，所以 KV cache 不等价
        # hash(
        #     hash(prefix_tokens) +
        #     token_ids
        # )
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())

        # intdigest() 是 xxhash 这个库提供的方法
        # 意思是：把当前算好的 hash 值，以 Python 整数 int 的形式返回
        return h.intdigest()

    # move this block to used list
    # 把编号为 block_id 的物理块, 从空闲状态切换成已使用状态
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # 只有 ref_count == 0 的物理块，才允许改变空闲/已使用状态
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    # 把编号为 block_id 的物理块, 从已使用状态切换成空闲状态
    def _deallocate_block(self, block_id: int) -> None:
        # 只有 ref_count == 0 的物理块，才允许改变空闲/已使用状态
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 判断是否能给序列分配 block
    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        # seq.num_blocks 这条序列当前 token 需要多少个 block
        # 如果 空闲block数量 >= 这条序列需要的 block数量 就说明可以分配
        return len(self.free_block_ids) >= seq.num_blocks

    # 给序列分配 block
    def allocate(self, seq: Sequence) -> None:
    # allocate(seq)
    #     初始化 h = -1，表示还没有前缀 hash

    #     遍历 seq 的每个逻辑 block:
    #         1. 取出当前逻辑 block 的 token_ids
    #         2. 如果是 full block:
    #                根据 前缀 hash + 当前 token_ids 计算 hash
    #            如果是 partial block:
    #                h = -1，不参与 prefix cache

    #         3. 用 h 去 hash_to_block_id 查物理 block_id

    #         4. 判断是否 cache hit:
    #                block_id 找到了
    #                并且 blocks[block_id].token_ids == token_ids

    #         5. 如果 cache hit:
    #                seq.num_cached_tokens += block_size
    #                如果该物理 block 正在 used 中:
    #                    ref_count += 1，表示多个 seq 共享
    #                否则:
    #                    把这个 block 重新标记为 used

    #         6. 如果 cache miss:
    #                从 free_block_ids 取一个空闲物理 block
    #                写入 hash 和 token_ids
    #                如果是 full block:
    #                    登记到 hash_to_block_id，供以后复用

    #         7. 把最终选中的物理 block_id 加入 seq.block_table


        h = -1
        # seq.num_blocks 这条序列当前 token 需要多少个 block
        # 遍历 seq 的所有 逻辑块
        for i in range(seq.num_blocks):
            no_cache_found = False
            
            # 取出这条序列的第 i 个 block 对应的 token_ids
            token_ids = seq.block(i)
            
            # only compute hash for full blocks, always -1 for partial blocks
            # 如果是 partial block
            # 也就是最后一个没填满的 block
            # 就不给有效 hash
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # if cache miss or hash collision
            # 1. cache miss 在 hash_to_block_id 里找不到这个 hash
            # 2. hash collision 找到了这个 hash 但是对应的 token_ids 不一样
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # update sequence information
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                # 存在这种情况
                # 这个物理 block 目前没有被任何正在运行的 Sequence 使用
                # 但是 这个 block 里的 kv cache 内容还在
                if block_id not in self.used_block_ids:
                    # 不在 used 集合里 则重新分配这个block
                    block = self._allocate_block(block_id)
                else:
                    # update block information
                    # 如果 block 当前已经在使用 就增加引用计数
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache miss
                # 从 free_block_ids 里分配一个 block
                # 写入 hash 和 token
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                # 如果是完整的 block 则登记到 hash_to_block_id
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            # 更新 逻辑块 到 物理块 的映射
            # 列表下标 = 逻辑 block 编号
            # 列表里的值 = 物理 block id
            seq.block_table.append(block.block_id)
    
    # deallocate 的目的
    # 一条 Sequence 结束后，把它占用的物理 KV cache block 释放掉
    # 如果某些 block 还被别的序列共享，就不能真正释放
    def deallocate(self, seq: Sequence) -> None:
        # Logic:
        # 1. 遍历 seq.block_table
        # 2. 每个 block 的 ref_count -= 1
        # 3. 如果引用计数变成 0 , 就真正释放
        # 4. 清空序列自己的 block_table
        # 5. 清空 num_cached_tokens。

        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # update sequence information
        # 该 seq 已经结束
        # 清空它的 block_table 和 num_cached_tokens
        seq.block_table = []
        seq.num_cached_tokens = 0


    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    # 用于 decode 阶段
    # 每生成一个 token 后，序列长度会增加
    # 有时候新 token 可以放进已有的最后一个 partial block
    # 有时候会刚好需要新 block
    def can_append(self, seq: Sequence) -> bool:
        # 如果当前 token 数量正好是 block_size 的整数倍
        # 说明 block 已经满了
        # 下次再 append token 时就需要新 block
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    # 追加 token 后更新 block
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        # 该 seq 的最后一个 block 的 id
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        # 分为 3 种情况
        # 1. 最后一个 block 刚好变满
        if seq.num_tokens % self.block_size == 0:
            # 计算这个 full block 的 hash
            # 更新最后一个物理 block 的 hash 和 token
            # 把它加入 hash_to_block_id, 以后可以被 prefix cache 复用
            h = self.compute_hash(
                token_ids = seq.block(seq.num_blocks - 1), 
                prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash
            )
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # if one new block is needed
        # 2. 新 token 开启一个新 block
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            # 检查前一个 block 已经是 full block
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # else, do nothing
        # 3. 还在当前 partial block 里
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
