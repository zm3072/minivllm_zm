import torch.nn as nn 
import torch
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        # tp_dim 切分维度
        # tp_rank 当前是哪张卡
        # tp_world_size 总共有多少张卡
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        
        # self.weight 不是“普通数值”，而是一个 Python 对象：nn.Parameter
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
# 等效于最基本的线性层
class ReplicatedLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter

# 切分的结果是最终结果的一部分，不需要通讯合并
# 完整答案的一段
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        # 在 super().init(...) 里，括号中的内容就是“传给父类构造函数的参数”
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
    
        # narrow: 在第 dim 个维度上，从 start 位置开始，连续取 length 个元素。
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# an extension of ColumnParallelLinear by merging several matrices
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        # loaded_weight_id 管“放哪儿”
        # tp_rank 管“拿哪片”
        
        # 详细版本
        # 先根据 loaded_weight_id 确定当前子权重在本地合并参数中的目标区间；
        # 再根据 tp_rank 从 checkpoint 的完整子权重中切出当前 GPU 负责的 shard；
        # 最后把这个 shard 写入本地参数的目标区间。

        param_data = param.data
        # compute offset 
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        # 
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Calculate per-GPU output size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    # load_weight_id: q, k, v
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        # batch_size * num_heads * num_token * head_size
        param_data = param.data
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # compute offset

        # loaded_weight_id 管“放哪儿”
        # tp_rank 管“拿哪片”

        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


# 是“完整答案的一部分贡献”
# 需要all_reduce把每张卡的部分结果加起来才能得到完整答案
# 所以需要通信
class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        
        # 逻辑上这里的 x 也必须已经被切分了
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the input row


        full_data_input_size = loaded_weights.size(1)
        # dim size after sharding
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


if __name__ == "__main__":
    # Example usage
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)



# all_reduce
# 大家手里都有一份同形状的局部结果，然后做一次聚合
# 比如 SUM / MIN / MAX，最后每个 GPU 都拿到聚合后的结果

# rank0: y0
# rank1: y1
# rank2: y2

# all_reduce(SUM)

# rank0: y0 + y1 + y2
# rank1: y0 + y1 + y2
# rank2: y0 + y1 + y2


#---------------------------------------------------------


# all_gather
# 大家手里各有一份不同片段，不做求和，只是把所有片段收集起来
# 最后每个 GPU 都拿到完整拼接结果

# rank0: part0
# rank1: part1
# rank2: part2

# all_gather

# rank0: [part0, part1, part2]
# rank1: [part0, part1, part2]
# rank2: [part0, part1, part2]