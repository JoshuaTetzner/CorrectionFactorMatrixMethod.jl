# General Usage

The recommended entry point is [`CFMM.assemble`](@ref). It accepts a boundary
integral operator and its test and trial spaces, constructs an optimized
`H2Trees.BlockTree`, and returns a [`PetrovGalerkinCFMM`](@ref).

```julia
matrix = CFMM.assemble(operator, testspace, trialspace)
```

The lower-level `PetrovGalerkinCFMM` constructor accepts an existing tree.
Keywords control near and far quadrature, scheduling, and whether a separate
transpose FMM is built.

The default FMM configuration uses ExaFMMt with expansion order `p = 8` and a
critical leaf size of `ncrit = 50`.

```@example configuration
using CorrectionFactorMatrixMethod

configuration = CorrectionFactorMatrixMethod.FMMFunctor(; p=10, ncrit=80)
(p=configuration.p, ncrit=configuration.ncrit)
```

The resulting operator implements the `LinearMaps.jl` interface, so it can be
multiplied by vectors without assembling the dense boundary-element matrix.
