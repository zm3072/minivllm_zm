from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        # KV cache 每个block能容纳多少token
        self.block_size = block_size # number of tokens per block
        # record sequence id
        self.seq_id = next(Sequence.counter)
        # status
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        self.token_ids = copy(token_ids)
        # last token
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens
        # 初始化时 num_tokens = num_prompt_tokens
        # 后续每生成一个token num_tokens += 1, num_prompt_tokens不变
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        # num_cached_tokens = 0
        # 已经放入 KV cache 的 token 数
        self.num_cached_tokens = 0
        # block_table
        # 这条序列对应使用过的物理 cache block 编号
        self.block_table = []
        # sampling_params' related things
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    # 是否生成完成
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    # 已经生成了多少新 token
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # 返回 prompt 部分
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 返回 生成内容 部分
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 已经缓存的 token 占用了多少个 Block
    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    # 当前序列需要多少 Block
    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))
    
    # 最后一个 Block 里有多少个 token
    @property
    def last_block_num_tokens(self):
        full_blocks = int(math.floor(self.num_tokens / self.block_size))
        return len(self.token_ids[full_blocks * self.block_size : ])

    # 按 Block 取 token
    # 这个方法返回第 i 个 block 对应的 token
    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    # 追加生成 token
    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    def __getstate__(self):
        return (
            self.num_tokens, 
            self.num_prompt_tokens, 
            self.num_cached_tokens, 
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state
        # Check if this is prefill (num_completion_tokens == 0) or decode phase
        # 区分 prefill 和 decode 阶段
        # prefill 阶段保存完整的 token_ids
        # decode 阶段只保存最后一个 token
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            # Prefill: last_token_or_ids is the full token_ids list
            # last_token_or_ids 是完整的 token_ids 列表
            self.token_ids = last_token_or_ids
        else:
            # Decode: last_token_or_ids is just the last token
            # last_token_or_ids 是最后一个 token
            self.token_ids = [last_token_or_ids]
        # Restore last_token attribute
        self.last_token = self.token_ids[-1] if self.token_ids else None