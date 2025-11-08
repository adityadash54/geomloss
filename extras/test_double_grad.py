"""
Test if double_grad + KeOps causes issues
"""

import torch
from geomloss.kernel_samples import double_grad, energy_kernel, laplacian_kernel, gaussian_kernel

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

x = torch.randn((2, 10, 2), device=device)
y = torch.randn((2, 15, 2), device=device)

print("Testing kernels with and without double_grad:")
print("=" * 80)

kernels_to_test = [
    ("gaussian", gaussian_kernel),
    ("laplacian", laplacian_kernel),
    ("energy", energy_kernel),
]

for name, kernel_func in kernels_to_test:
    print(f"\n{name.upper()}:")
    
    # Without double_grad, without KeOps
    K1 = kernel_func(x, y, blur=0.5, use_keops=False)
    print(f"  No double_grad, no KeOps: mean={K1.mean().item():.6f}, NaN={torch.isnan(K1).any()}")
    
    # With double_grad, without KeOps
    K2 = kernel_func(double_grad(x), y, blur=0.5, use_keops=False)
    print(f"  With double_grad, no KeOps: mean={K2.mean().item():.6f}, NaN={torch.isnan(K2).any()}")
    
    # Without double_grad, with KeOps
    try:
        K3 = kernel_func(x, y, blur=0.5, use_keops=True)
        K3_val = K3.sum() if not isinstance(K3, torch.Tensor) else K3
        print(f"  No double_grad, with KeOps: type={type(K3)}, NaN={torch.isnan(K3_val).any() if isinstance(K3_val, torch.Tensor) else torch.isnan(K3_val)}")
    except Exception as e:
        print(f"  No double_grad, with KeOps: ERROR - {str(e)[:50]}")
    
    # With double_grad, with KeOps
    try:
        K4 = kernel_func(double_grad(x), y, blur=0.5, use_keops=True)
        K4_val = K4.sum() if not isinstance(K4, torch.Tensor) else K4
        print(f"  With double_grad, with KeOps: type={type(K4)}, NaN={torch.isnan(K4_val).any() if isinstance(K4_val, torch.Tensor) else torch.isnan(K4_val)}")
    except Exception as e:
        print(f"  With double_grad, with KeOps: ERROR - {str(e)[:50]}")
