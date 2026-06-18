import torch
import time 

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # Use nn.Parameter to make gamma learnable and loadable from checkpoints
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        self.eps = eps

    # 把一个方法伪装成"属性"来访问
    # layer.gamma 就像访问属性一样访问方法
    # layer = LayerNorm(gamma=torch.ones(4096))
    # layer.weight == layer.gamma

    # 代码意义就是给 self.weight 起了一个旧名字 gamma 为了兼容旧代码和数学公式
    # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
    @property
    def gamma(self):
        """Backward compatibility: gamma alias for weight"""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ

        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm
    
    # 参考Qwen3模型架构来理解
    # 这里是把残差相加的操作融合到 RMSNorm 里了，减少一次内存访问和计算
    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x + residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 刚刚进入layer的时候，没有上一层的output
        # 所以自然没有残差
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(8,4000,8000).cuda()
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x,fill_value=1)

    for _ in range(10): # Warm-up iterations
        _ = layer(x)
    
    # Without residuals
    times = [] 
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    # With residuals
    times.clear()
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    
