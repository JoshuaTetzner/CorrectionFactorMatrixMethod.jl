# Application Example

The FMM wrapper follows the `LinearMaps.jl` interface. This small stand-in for
an FMM backend is executed when the documentation is built:

```@example matrix-free
using CorrectionFactorMatrixMethod

dense_backend = [2.0 1.0; -1.0 3.0]
fmm = CorrectionFactorMatrixMethod.FMM(dense_backend)

x = [1.0, 2.0]
y = fmm * x

(size=size(fmm), result=y)
```

In a boundary-element application,
[`PetrovGalerkinCFMM`](@ref) wraps an ExaFMMt backend and adds the sparse near
correction. See [Package Extensions](@ref) for the packages that activate that
integration.
