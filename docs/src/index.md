# CorrectionFactorMatrixMethod.jl

CorrectionFactorMatrixMethod.jl builds matrix-free boundary-element operators
that combine a fast multipole approximation of far interactions with directly
assembled near-field corrections.

The package keeps the generic correction and linear-map machinery in its core.
Optional integrations with
[BEAST.jl](https://github.com/krcools/BEAST.jl),
[CompScienceMeshes.jl](https://github.com/krcools/CompScienceMeshes.jl), and
[ExaFMMt.jl](https://github.com/JoshuaTetzner/ExaFMMt.jl) are loaded through
Julia package extensions.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JoshuaTetzner/CorrectionFactorMatrixMethod.jl")
```

The current integration requires Julia 1.10 or later.
