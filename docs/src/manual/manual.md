# General Usage

The main entry point is
[`PetrovGalerkinCFMM`](@ref). It accepts a boundary integral operator, test and
trial spaces, and an `H2Trees.BlockTree`. Keyword arguments control near and far
quadrature, task parallelism, and whether a separate transpose FMM is built.

The default FMM configuration uses ExaFMMt with expansion order `p = 8` and a
critical leaf size of `ncrit = 50`.

```@example configuration
using CorrectionFactorMatrixMethod

configuration = CorrectionFactorMatrixMethod.FMMFunctor(; p=10, ncrit=80)
(p=configuration.p, ncrit=configuration.ncrit)
```

The resulting operator implements the `LinearMaps.jl` interface, so it can be
multiplied by vectors without assembling the dense boundary-element matrix.
