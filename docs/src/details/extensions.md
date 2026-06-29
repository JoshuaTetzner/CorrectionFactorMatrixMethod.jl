# Package Extensions

CorrectionFactorMatrixMethod.jl uses package extensions to keep optional
boundary-element and FMM dependencies out of the core load path.

`CFMMBEAST` loads when `BEAST` and `CompScienceMeshes` are available. It
provides quadrature, collocation, corrected-kernel assembly, and supported
Helmholtz and Maxwell operator methods.

`CFMMExaFMMt` loads when `ExaFMMt` and `BEAST` are available. It maps BEAST
operator parameters to ExaFMMt options and constructs the FMM maps used by
[`PetrovGalerkinCFMM`](@ref).

Load all three optional packages before constructing a supported
boundary-element operator:

```julia
using BEAST, CompScienceMeshes, ExaFMMt
using CorrectionFactorMatrixMethod
```
