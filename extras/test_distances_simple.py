"""
Simple test to isolate the energy/laplacian issue
"""

import torch
from geomloss.utils import distances, squared_distances

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# Small test case
x = torch.randn((2, 10, 2), device=device)
y = torch.randn((2, 15, 2), device=device)

print("Testing distances function:")
print("=" * 60)

# Test without KeOps
D_no_keops = distances(x, y, use_keops=False)
print(f"Without KeOps: shape={D_no_keops.shape}, mean={D_no_keops.mean().item():.6f}")
print(f"  has NaN: {torch.isnan(D_no_keops).any()}")
print(f"  has Inf: {torch.isinf(D_no_keops).any()}")

# Test with KeOps
try:
    D_with_keops = distances(x, y, use_keops=True)
    print(f"\nWith KeOps: type={type(D_with_keops)}")
    
    # Try to evaluate it
    D_eval = D_with_keops.sum()
    print(f"  Sum: {D_eval.item():.6f}")
    print(f"  has NaN: {torch.isnan(D_eval)}")
    print(f"  has Inf: {torch.isinf(D_eval)}")
    
except Exception as e:
    print(f"\nWith KeOps: ERROR - {e}")

print("\n" + "=" * 60)
print("Testing exp(-distances) for Laplacian:")
print("=" * 60)

blur = 0.5

# Without KeOps
D = distances(x / blur, y / blur, use_keops=False)
K = torch.exp(-D)
print(f"Without KeOps: mean={K.mean().item():.6f}")
print(f"  has NaN: {torch.isnan(K).any()}")

# With KeOps
try:
    D_keops = distances(x / blur, y / blur, use_keops=True)
    K_keops = (-D_keops).exp()
    K_sum = K_keops.sum()
    print(f"\nWith KeOps: sum={K_sum.item():.6f}")
    print(f"  has NaN: {torch.isnan(K_sum)}")
except Exception as e:
    print(f"\nWith KeOps: ERROR - {e}")

print("\n" + "=" * 60)
print("Testing matrix multiplication:")
print("=" * 60)

# Create weights
alpha = torch.ones((2, 10), device=device) / 10
beta = torch.ones((2, 15), device=device) / 15

# Without KeOps
D = distances(x, y, use_keops=False)
K = -D
result = (K @ beta.unsqueeze(-1)).squeeze(-1)
print(f"Without KeOps: result mean={result.mean().item():.6f}")

# With KeOps  
try:
    D_keops = distances(x, y, use_keops=True)
    K_keops = -D_keops
    result_keops = (K_keops @ beta.unsqueeze(-1)).squeeze(-1)
    print(f"With KeOps: result mean={result_keops.mean().item():.6f}")
    print(f"  has NaN: {torch.isnan(result_keops).any()}")
except Exception as e:
    print(f"With KeOps: ERROR - {e}")
