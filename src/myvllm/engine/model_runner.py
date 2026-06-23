import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *


# Sequence:
#     记录一条请求的 token、block_table、状态

# BlockManager:
#     给 Sequence 分配物理 KV cache block

# ModelRunner:
#     根据 Sequence 的 token 和 block_table，真正调用模型 forward

class ModelRunner:
    def __init__(
            self, 
            config: dict, # 模型和运行配置
            rank: int,    # 当前进程/ GPU 编号
            event: Event | list[Event] # 用于多进程通信同步
        ):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        
        # 这部分完全没懂

        # 初始化分布式通信。
        # nccl 是 GPU 间通信后端。
        # torch.cuda.set_device(rank) 让当前进程绑定到对应 GPU。

        # 这个通信组建立后，多个进程就可以通过 NCCL 做 GPU 通信
        self.rank = rank
        dist.init_process_group(
            'nccl', 
            "tcp://localhost:12345", 
            world_size=config['world_size'], 
            rank=rank
        )
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        # 如果配置里有模型路径
        # 那就从 checkpoint 加载预训练权重
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        # 采样层 用 logits 采样下一个 token

        # Sampler层的细节没仔细看 一会补一下
        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        # 保存当前默认 dtype
        # 后面计算 KV cache 大小时需要用到 dtype 的字节数
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        # 调试标志 当前代码里没有实际使用
        self._first_decode = False

        # warm up model so that we know peak memory usage
        # 先跑一次大输入，测出模型峰值显存
        self.warmup_model()

        # allocate kv cache
        # 根据剩余显存分配 KV cache
        self.allocate_kv_cache()


        # capture cuda graph for decoding
        # 完全不懂 后续看一下
        
        # 如果没有强制 eager 就捕获 decode 阶段的 CUDA graph 加速固定形状 decode。
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            # 同步所有 rank
            # 确保都初始化完模型和 KV cache
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                # 尝试清理旧共享内存
                # 啥时候出现的旧共享内存？
                # SharedMemory用法是什么？
                # .close()和.unlink()分别是啥？
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()

                # 打开 rank0 创建好的共享内存
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__


    # 因为多 GPU 下，每个 GPU 是一个独立进程
    # rank 0 收到请求后，不能只自己跑模型
    # 所以需要通过共享内存和事件机制，将任务分发给其他 GPU（worker）进程
    # only use read when rank != 0
    def read_shm(self):
        # 这个函数只能在多 GPU / 多进程时，并且只能由非 rank 0 的进程调用
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        
        # worker 在这里等待 rank0 发信号
        self.event.wait()

        # share memory layout:
        # 前 4 字节：后面数据的长度 n
        # 后 n 字节：pickle 序列化后的真实命令
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])

        # 这条命令我已经读完了
        # 下一次还要继续等待 rank 0 重新 event.set()
        # 如果不 clear()，下一次循环就会直接跳过 wait()，导致重复执行同一条命令
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        # 这个函数只能在多 GPU / 多进程时，由 rank 0 调用
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)

        # 因为共享内存本身只是一块 bytes 区域，它不知道你写的数据有多长，所以需要手动在前面写长度
        # method_name = "run" args = (seqs, True)
        # (method_name, args) -> ("run", (seqs, True))
        # (method_name, *args) -> ("run", (eqs, True))

        # 下面的整个过程就是写入共享内存
        data = pickle.dumps((method_name, *args))
        n = len(data)

        # share memory layout:
        # 前 4 字节：后面数据的长度 n
        # 后 n 字节：pickle 序列化后的真实命令
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        
        # 通知每一个 worker
        # 共享内存里有新命令了，你可以从 wait() 醒来读取了
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    # 清理资源
    def exit(self):
        if self.world_size > 1:
            # 关闭当前进程对共享内存的访问句柄
            self.shm.close()
            if self.rank == 0:
                # 删除系统里的共享内存对象
                self.shm.unlink()

        # 代码里的逻辑 只要 enforce_eager 为 False
        # 初始化阶段就会尝试创建 CUDA graph
        # 所以没有 eager 模式 就直接删掉 CUDA graph 相关的变量
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars 

        # 清理 Pytorch 分布式通信环境
        # 确认真的已经初始化过 process group
        if dist.is_initialized():
            # 真初始化过 再进行销毁
            # 初始化时 会创建一些底层资源
            # 如果不用 需要进行销毁
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    # 让非 rank 0 的 worker 进程一直待命，等待 rank 0 通过共享内存发命令；
    # 一旦收到命令，就执行对应方法；
    # 如果收到 exit，就清理资源并退出循环。
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        # 不会一直循环下去
        # 执行完当前的命令后 事件信号会被 clear
        # 然后 卡在 wait() 里 等待 rank 0 发信号
        # 等到 rank0 发信号后 通过set() 唤醒 wait()，继续循环
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            # 遇到 exit 时才退出循环
            if method_name == 'exit':
                self.exit()
                break


    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    # 用统一入口根据字符串方法名调用 ModelRunner 上的实际方法
    # 如果当前是 rank 0，还会先把这条调用命令广播给 worker
    def call(self, method_name: str, *args: dict):

        # 如果是 rank 0，而且还有 worker
        # 那 rank 0 在自己执行方法前，要先通知 worker 一起执行
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            # rank 0 把这次调用写到共享内存，并通过 event 唤醒 worker
            self.write_shm(method_name, args)
        
        # getattr() 是 Python 内置函数，用来根据字符串取对象属性
        # method_name = "run"
        # method = getattr(self, "run", None)
        # method = self.run
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory

    # 在正式分配 KV cache 之前
    # 先用一批假的长序列跑一次模型 prefill
    # 触发模型执行并测出模型 forward 的峰值显存占用
    def warmup_model(self):
        
        # 1. 清理 Pytorch CUDA 缓存池里暂时不用的显存
        torch.cuda.empty_cache()

        # 2. 重置 CUDA 的峰值显存统计
        torch.cuda.reset_peak_memory_stats()

        # 3. 从配置里读取一个 batch 最多允许包含多少 token
        max_tokens = self.config['max_num_batch_tokens']
        # 3. 读取单条序列的最大长度
        max_model_length = self.config['max_model_length']

        # 4. 估算 warmup 时要构造多少条最大长度序列
        # 就是为了模拟“单条序列最长”的压力
        # max_tokens = 8192
        # max_model_length = 2048
        # batch_size = 8192 // 2048 = 4
        # 构造 4 条长度 2048 的序列，总 token 数正好接近最大 batch token 数
        batch_size = max_tokens // max_model_length

        # 5. 构造一批假的 Sequence 每条序列的 token 都是 0
        # 因为 warmup 的目的不是生成正确结果，而是让模型跑一遍，触发 kernel 和显存分配
        # 所以用 0 就可以
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        
        # 6. 调用 run() 执行 prefill forward
        # 因为 prefill 会处理较长 prompt
        # 一次性计算大量 token 更容易触发接近真实高负载的显存占用
        self.run(seqs, is_prefill=True)

        # 7. 再次清理 CUDA 缓存
        # 这一步尽量把不用的显存释放掉
        # 为后面真正分配 KV cache 做准备
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    # 根据当前 GPU 剩余显存 计算最多能放多少个 KV cache 物理块
    # 然后一次性分配一个大的 KV cache 池
    # 并把每一层 Attention 的 k_cache / v_cache 指向这个池


    # 3. 算一个 KV cache block 需要多少字节
    # 4. 算最多能分配多少个 block
    # 5. 多 GPU 时取所有 rank 中最小的 block 数
    # 6. 一次性分配大 KV cache tensor
    # 7. 把每层 Attention 的 k_cache / v_cache 指向对应切片
    def allocate_kv_cache(self):
        # find all available memory
        # 1. 看 GPU 还有多少显存
        # free_mem: 当前空闲显存，单位 byte 
        # total_mem: GPU 总显存，单位 byte
        free_mem, total_mem = torch.cuda.mem_get_info()
        
        # 不是把所有空闲显存都拿来用，而是只用其中一部分
        # 留一点安全余量，避免把 GPU 显存吃满
        total_free_mem = free_mem * self.config['gpu_memory_utilization']

        # 读取之前 warmup 阶段记录到的 峰值显存占用
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']

        # 计算当前已经占用的显存
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        

        # 2. 根据 warmup 的峰值显存，预留模型运行时临时显存
        # peak_mem_usage - current_mem_usage
        # 模型 forward 运行时，可能额外需要的临时显存
        
        # 计算当前可用显存 = 总空闲显存 - (峰值显存占用 - 当前显存占用)
        # 要减去 这部分 可能需要的临时显存，避免分配 KV cache 后模型 forward 时 OOM
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        # 读取模型层数
        num_layers = self.config['num_layers']
        
        # 计算当前 rank 上负责的 KV head 数
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        
        # 计算每个 head 的维度
        # 如果有就直接读取 否则就计算
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block

        # 3. 计算一个 KV cache block 需要多少字节
            # self.block_size: 每个 block 的 token 数
            # 2: 因为 KV cache 有 k_cache 和 v_cache
            # num_layers: 每一层都有 KV cache
            # num_kv_heads: 当前 rank 上的 KV head 数
            # head_dim: 每个 head 的维度
            # self.default_dtype.itemsize: 当前 dtype 的字节数
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        
        # 4. 计算当前 GPU 上最多能分配多少个 KV cache block
        # 用可用显存除以单个 block 的大小，得到最多能放多少个 KV cache block
        num_available_kv_blocks = int(available_mem // block_bytes)

        # 5. 检查至少能放 1 个 block
        # 如果连一个 KV block 都放不下，说明显存不够，直接报错
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.

        # 如果是多 GPU / 多 rank 模式
        # 需要同步所有 rank 的可用 block 数
        # 因为不同 rank 的空闲显存可能不一样
        # 比如 rank 0 可能多了调度器、NCCL buffer、主进程开销，所以它可用显存更少
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            
            # 当前 rank 的 block 数变成 CUDA tensor
            # 因为 dist.all_reduce() 操作的是 tensor 不是普通 Python int
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            
            # 让所有 rank 交换自己的 num_available_kv_blocks，然后取最小值
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            
            # 单 GPU 就直接本地算出 Block 数
            self.config['max_cached_blocks'] = num_available_kv_blocks
        
        # 只让 rank 0 打印最终全局 block 数，避免所有 rank 重复打印
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values

        # 6. 一次性分配一个大的 KV cache tensor
        # [2, num_layers, max_cached_blocks, block_size, num_kv_heads, head_dim]
        allocated_kv_cache = torch.zeros(
            2, 
            self.config['num_layers'], 
            self.config['max_cached_blocks'], 
            self.block_size, 
            num_kv_heads, 
            head_dim, 
            device=f'cuda:{self.rank}'
        )
        
        # 准备一个层编号
        # 后面遍历模型模块时 遇到一个 attention 层 就给它分配对应的第 layer_id 层 KV cache
        layer_id = 0

        # 遍历模型里的所有子模块
        for module in self.model.modules():
            # 判断这个 module 是不是有 k_cache 和 v_cache 属性
            # 一般 只有 attention 模块才有
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1
    # 最终效果：
    # 模型每一层 Attention 都拿到自己的 k_cache/v_cache
    # 这些 cache 都来自同一个大 tensor allocated_kv_cache 的不同切片


    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )