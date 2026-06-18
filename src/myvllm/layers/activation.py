import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

# SiLU gate 激活层
# 参考Qwen3模型架构来理解
# 这里是把 SiLU激活 和 后续的乘法操作 融合到一个层里了，减少一次内存访问和计算
# SiLU(x) = x * sigmoid(x) 相当于算了一个gate
class SiluAndMul(nn.Module):
    """
    A custom activation layer that applies the SiLU (Sigmoid Linear Unit) activation
    function followed by element-wise multiplication with the input tensor.
    """

    def __init__(self):
        super().__init__()
    
    # 性能优化
    # torch.compile 也可以直接编译 nn.Module 或者函数
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

if __name__ == "__main__":
    # Example usage
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()  # Example input tensor with shape (400, 800)
    
    for _ in range(10):  # Warm-up iterations
        _ = layer(input_tensor)

    times = []
    for _ in range(100):  # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
