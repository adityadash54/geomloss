"""
Detailed trace of where NaN appears in energy/laplacian
"""

import torch
import warnings
warnings.filterwarnings('error')  # Turn warnings into errors to see where they come from

from geomloss import SamplesLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

x = torch.randn((3, 100, 2), device=device)
y = torch.randn((3, 150, 2), device=device)

print("Testing with detailed error tracking:")
print("=" * 80)

for metric in ["gaussian", "laplacian", "energy"]:
    print(f"\n{metric.upper()}:")
    try:
        L = SamplesLoss(metric, blur=0.5, backend="online")
        result = L(x, y)
        print(f"  Result: {result}")
        print(f"  Mean: {result.mean().item():.6f}")
        print(f"  Has NaN: {torch.isnan(result).any()}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
