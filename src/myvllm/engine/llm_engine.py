import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


# 在 worker 进程里初始化一个属于该 rank/GPU 的 ModelRunner
# 然后进入 loop() 等待 rank 0 发命令
def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    # 多进程里，子进程输出经常会被缓冲
    # 这几行是为了让 worker 进程里的 print()、报错输出更及时
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    # 创建 worker 自己的 ModelRunner
    model_runner = ModelRunner(config, rank, event)

    # 进入 loop，等待 rank 0 指令
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        # 模型配置              
        self.config = config

        world_size = config.get("world_size", 1)
        
        # 拿到一个固定使用 spawn 启动方式的多进程上下文
        # 让每个 worker 进程都以 spawn 方式启动
        # spawn 方式的启动方式是安全的，适合在多 GPU 上运行
        # 让每个 worker rank 在干净的新 Python 进程里独立初始化 CUDA、NCCL 和模型
        # 避免 fork 继承父进程 CUDA 状态带来的卡死或错误
        ctx = mp.get_context("spawn")

        # 保存所有 worker 进程的 Process 对象
        self.processes = []
        # 每个 worker 对应的 event 对象
        self.events = []

        # rank0 就是当前主进程
        # 从 rank1 开始创建 worker
        for i in range(1, world_size):
            # 为当前 worker 创建一个事件信号
            event = ctx.Event()
            
            # 创建一个子进程对象
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        
        # start the engine only on the master thread with rank = 0
        # 在当前主进程里创建 rank0 的 ModelRunner
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        # 多 GPU 时 前面的 worker 进程也各自创建了自己的 ModelRunner

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.

        # 创建调度器
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )

        # 退出程序时 自动调用 self.exit() 清理资源
        atexit.register(self.exit)

    # 通知所有模型执行进程退出，并等待 worker 子进程真正结束
    def exit(self):
        # 给所有 worker 进程发 exit 命令
        # 等价于 self.model_runner.exit()
        self.model_runner.call("exit")

        # 删掉 LLMEngine 对 rank0 的 ModelRunner 引用   
        del self.model_runner

        # 遍历所有的 worker 进程，通知它们退出
        for process in self.processes:
            # join() 阻塞当前进程，直到这个 Process 对象对应的子进程结束
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    
    # 执行一轮调度 + 一轮模型 forward + 一轮后处理
    def step(self) -> tuple[list[int], bool]:
        # 调用调度器 决定这一轮要跑什么
        scheduled_sequences, is_prefill = self.scheduler.schedule()

        # 如果没有任何 sequence 被调度，就直接返回空结果
        if not scheduled_sequences:
            return [], is_prefill
        
        # run the model
        # 调用模型执行器的 run() 方法，传入本轮调度的 sequence 列表和是否是 prefill
        # 这里不用 self.model_runner.run()，而是用 self.model_runner.call("run", ...)
        # 是要考虑到多 GPU 的情况，rank0 进程会把 run() 调用广播给所有 worker rank

        # rank 0:
        # call("run", scheduled_sequences, is_prefill)
        #     -> 写 shared memory
        #     -> event.set() 唤醒 rank1/rank2/...
        #     -> rank0 自己也执行 run()

        # worker rank:
        # loop() 等待 event
        # -> 收到 "run"
        # -> 调用自己的 run(scheduled_sequences, is_prefill)
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)

        # Move outputs to CPU and convert them to a list
        # 移动到 CPU，再转成 Python list
        if outputs is not None:
            outputs = outputs.cpu().tolist()

        # postprocess the outputs
        #把模型生成的新 token 交给 scheduler 做后处理
        # scheduler.postprocess 模型跑完后的更新
        self.scheduler.postprocess(scheduled_sequences, outputs)

        # 这里的 outputs 是一个 list
        # 里面每个元素是一个 tuple (seq_id, completion_token_ids) 
        # completion_token_ids 只有模型生成的 token id 不包含 prompt
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
    
        # 如果是 prefill，num_processed_tokens 是所有 scheduled_sequences 的 token 数量之和
        # 如果是 decode，num_processed_tokens 是 scheduled_sequences 的数量
        # 因为 decode 每个 sequence 只生成一个 token
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        # 返回结果
        # outputs
        # 本轮刚刚完成的 sequence 的结果
        # 格式是 [(seq_id, completion_token_ids), ...]

        # num_processed_tokens
        # 本轮处理 token 数，用于统计 tokens/sec

        # is_prefill
        # 本轮是 prefill 还是 decode，用于区分打印 prefilling/decoding 吞吐
        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    # 把用户输入的一条文本 prompt
    # 转换成引擎内部能调度的 Sequence
    # 然后放进 scheduler 的 waiting 队列

    # 原始 prompt 和 生成参数 (temperature max_tokens ignore_eos max_model_length)
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        self.scheduler.add_sequence(
            Sequence(
                token_ids=self.tokenizer.encode(prompt), 
                block_size=self.config['block_size'],
                sampling_params=sampling_params
            )
        )

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    
    # 接收一批 prompts 把它们加入调度器
    # 然后不断调用 step() 推进推理 直到所有请求生成结束
    # 最后把 token ids 解码成文本返回
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        # 把所有 prompt 加入 scheduler
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        
        # 这个字典用来保存已经完成的请求结果
        # 格式大概是 {seq_id: token_ids} {0: [100, 101, 102], 1: [200, 201]}
        generated_tokens = {}
        
        # 不断调用 step() 推进推理 直到所有请求生成结束
        # 只要 scheduler 里还有请求没完成 就继续循环
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()

            # 加上 1e-10 是为了避免极端情况下除以 0
            running_time = end_t - start_t + 1e-10

            # 根据不同的阶段打印吞吐
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            
            # 更新已经完成的请求结果
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        # 按照 seq_id 排序
        # 这是因为 batch 里的请求可能完成顺序不同
        # 短的 prompt 可能先完成，长的 prompt 可能后完成
        # 但 API 返回结果应该和用户传入 prompt 的顺序一致
        # 所以用 seq_id 排序把结果排回原来的提交顺序
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        
        # 把 token ids 解码成文本
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        
        # 最终返回 1.生成出来的文本 2.生成出来的 token ids
        return output
