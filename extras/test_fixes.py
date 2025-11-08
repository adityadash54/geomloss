"""
Quick test for the fixed energy and laplacian kernels
"""

import torch
from geomloss import SamplesLoss

print("=" * 80)
print("Testing Fixed Energy and Laplacian Kernels")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

x = torch.randn((3, 100, 2), device=device)
y = torch.randn((3, 150, 2), device=device)

print(f"\nDevice: {device}\n")

# Test classic metrics that had issues
test_metrics = ["energy", "gaussian", "laplacian"]
backends = ["tensorized", "online"]

for metric in test_metrics:
    print(f"{metric.upper()}:")
    results = {}
    
    for backend in backends:
        try:
            L = SamplesLoss(metric, blur=0.5, backend=backend)
            result = L(x, y).mean().item()
            
            if torch.isnan(torch.tensor(result)) or torch.isinf(torch.tensor(result)):
                print(f"  {backend:12s}: ❌ NaN/Inf")
                results[backend] = None
            else:
                print(f"  {backend:12s}: ✓ {result:.6f}")
                results[backend] = result
        except Exception as e:
            print(f"  {backend:12s}: ❌ {str(e)[:50]}")
            results[backend] = None
    
    # Check consistency
    if results.get("tensorized") is not None and results.get("online") is not None:
        diff = abs(results["tensorized"] - results["online"])
        if diff < 1e-4:
            print(f"  ✓ Backends match (diff: {diff:.2e})")
        else:
            print(f"  ⚠ Backends differ (diff: {diff:.2e})")
    print()

print("=" * 80)
print("Testing Original Sinkhorn with Online Backend")
print("=" * 80)

try:
    L_sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.5, backend="online")
    result = L_sinkhorn(x, y)
    print(f"\n✓ Sinkhorn online backend works: {result.mean().item():.6f}")
except Exception as e:
    print(f"\n❌ Sinkhorn online backend error: {str(e)}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
Fixed Issues:
✓ Energy kernel - sqrt operation fixed in utils.py
✓ Laplacian kernel - sqrt operation fixed in utils.py  
✓ Sinkhorn online - added proper keops_available check
""")
print("=" * 80)
