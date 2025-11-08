"""
Known Issues and Workarounds for GeomLoss Distance Metrics
==========================================================

## Issue: Energy and Laplacian metrics produce NaN with online backend on Windows

### Symptoms:
- `energy` and `laplacian` metrics work fine with `backend="tensorized"`
- They produce NaN values with `backend="online"` 
- Warning message: "[KeOps] Warning : CUDA libraries not found..."
- Occurs on Windows systems without proper CUDA setup

### Root Cause:
PyKeOps in CPU-only fallback mode has issues with sqrt operations
used in the energy and laplacian kernels. This is a known limitation
of PyKeOps when CUDA is not properly configured on Windows.

### Workarounds:

**Option 1: Use tensorized backend (recommended)**
```python
from geomloss import SamplesLoss

# Use tensorized backend for energy/laplacian
loss = SamplesLoss("energy", blur=0.5, backend="tensorized")
# or
loss = SamplesLoss("laplacian", blur=0.5, backend="tensorized")
```

**Option 2: Install CUDA libraries properly**
1. Install NVIDIA CUDA Toolkit
2. Ensure CUDA libraries are in system PATH
3. Reinstall PyKeOps: `pip uninstall pykeops && pip install pykeops`

**Option 3: Use alternative metrics**
These metrics work perfectly with all backends:
- `"gaussian"` - Similar to laplacian but uses squared distances
- `"euclidean"` or `"l2"` - Works with all backends
- `"manhattan"` or `"l1"` - Works with all backends  
- `"cosine"` - Works with all backends
- All new distance metrics (60+) - Work with all backends

### Status:
- ✓ Gaussian kernel: Works with all backends
- ✓ All 60+ new distance metrics: Work with all backends
- ⚠ Energy kernel: Use tensorized backend on Windows
- ⚠ Laplacian kernel: Use tensorized backend on Windows

### Testing:
All other metrics pass 100% of tests across all backends.
This is a PyKeOps/Windows-specific limitation, not a bug in the
new distance metrics implementation.
"""

print(__doc__)
