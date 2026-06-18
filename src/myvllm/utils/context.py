from dataclasses import dataclass 
import torch 

# @dataclass 是一个装饰器，用于简化类的定义，自动生成 __init__、__repr__、__eq__ 等方法。
# 自动生成初始化函数，否则我们需要自己写一个 __init__ 方法来初始化这些属性。
# class Context:
#     def __init__(self, is_prefill=False, cu_seqlens_q=None, ...):
#         self.is_prefill = is_prefill
#         self.cu_seqlens_q = cu_seqlens_q
#         ...
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
# torch.Tensor | None 意思为可以是一个 torch.Tensor 对象，也可以是 None

# 用于存储当前的上下文信息的全局变量
_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
