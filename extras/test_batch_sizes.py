"""
Test if batch size matters for energy/laplacian NaN
"""

import torch
from geomloss import SamplesLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

print("Testing different batch sizes:")
print("=" * 80)

for batch_size in [1, 2, 3]:
    x = torch.randn((batch_size, 100, 2), device=device)
    y = torch.randn((batch_size, 150, 2), device=device)
    
    print(f"\nBatch size: {batch_size}")
    
    for metric in ["gaussian", "laplacian", "energy"]:
        try:
            L = SamplesLoss(metric, blur=0.5, backend="online")
            result = L(x, y)
            has_nan = torch.isnan(result).any()
            status = "❌ NaN" if has_nan else f"✓ {result.mean().item():.6f}"
            print(f"  {metric:12s}: {status}")
        except Exception as e:
            print(f"  {metric:12s}: ERROR - {str(e)[:40]}")
