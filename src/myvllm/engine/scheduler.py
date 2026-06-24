from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager



# waiting 队列里的新请求 -> prefill
# running 队列里的已有请求 -> decode

# waiting:
#   还没开始 prefill 的请求
# running:
#   已经 prefill 完，正在逐 token decode 的请求


# 新请求来了 -> add_sequence() 放进 waiting
# 每轮调用 schedule()
#   -> 优先尝试从 waiting 取请求做 prefill
#   -> 如果没有 prefill，就从 running 取请求做 decode
# 模型跑完后 -> postprocess() 更新 sequence 状态
class Scheduler:
    def __init__(
            self, 
            max_num_sequences: int, 
            max_num_batched_tokens: int, 
            max_cached_blocks: int, 
            block_size: int, 
            eos: int
        ):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos

    # 判断是否所有 sequence 全部结束
    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    # 新请求入队
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)

    # 调度核心
    # list[Sequence] : 本轮要送进 ModelRunner 的 sequence 列表
    # bool : True 表示 prefill，False 表示 decode
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        # 如果没有 prefill 的请求，就从 running 取请求做 decode
        while self.running:
            # 每次取一个正在生成的 sequence，准备给它追加一个 token
            seq = self.running.popleft()

            # use can_append to check whether we can append one more token
            # 先检查 KV cache 还能不能 append

            # 不能 append 的情况
            if not self.block_manager.can_append(seq):
                if self.running:
                    # 如果当前 sequence 不能 append
                    # 就把它放回队列，并抢占最后一个 sequence
                    # 释放队尾 seq 的 KV cache
                    # 然后 while 循环继续
                    # 当前 seq 下一轮再被 popleft 出来尝试 append
            # 一个例子 如果 有共同 KV cache 的 A B C 三个 sequence
            # A B C 都在 running 队列里
            # 如果 A 和 C 有共同 KV cache，释放 C 不应该伤到 A
            # 因为共享 block 的 ref_count 不会归零，所以不会被真正回收
            # 只有 ref_count 归零的 block 才会被真正回收
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())

            # 如果 running 里没有别的 seq 就抢占当前这个 seq 
                else:
                    self.preempt(seq)
                    break
            # 可以 append 的情况
            else:
                # 如果当前 scheduled token 数量已经达到上限
                # 或者 scheduled sequence 数量已经达到上限
                # 就把当前 seq 放回 running 队列，并结束调度
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break

                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                # 追加一个 token 的空间
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        # 重新放回 并且 保持原顺序
        if scheduled_sequences:
            # reversed 是为了保持原顺序
            # running 是 deque，pop出来的顺序是从左到右的
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False

    # preempt() 抢占
    # 释放这个 sequence 当前占用的 KV cache block
    # 状态改回 WAITING
    # 放回 waiting 队首
    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    # 模型跑完后的更新
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            # 把新 token 追加到 sequence 里
            seq.append_token(token_id)

            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)

            # 检查是否该结束
            # 1. 生成了 EOS
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            
            # 2. 达到 max_tokens      
            # 限制“最多新生成多少”     
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            
            # 3. 达到 max_model_length
            # 限制“prompt + 生成结果的总长度最多多少”           
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            # 如果结束了
            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                # 状态改为 FINISHED
                seq.status = SequenceStatus.FINISHED
                # 释放 KV cache block
                self.block_manager.deallocate(seq)
                # 从 running 队列移除
                self.running.remove(seq)