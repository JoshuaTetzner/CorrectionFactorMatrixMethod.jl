# Examples

The examples use BEAST for the boundary-element spaces and operators, ExaFMMt
for the far interactions, and `Krylov.gmres` for the matrix-free solve. Install
the example environment and run either script from the package root:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/efie.jl
julia --project=examples examples/mfie.jl
```

Both scripts write an interactive HTML plot containing the far-field pattern,
the electric field in the ``yz`` plane, and the surface-current magnitude.
They use the following packages:

```julia
using BEAST
using CompScienceMeshes
using CorrectionFactorMatrixMethod
using ExaFMMt
using Krylov
using LinearAlgebra
using PlotlyJS
```

## EFIE

The electric-field integral equation uses the Maxwell single-layer operator
with the same Raviart--Thomas space for testing and trial functions:

```julia
mesh = meshsphere(1.0, 0.4)
space = raviartthomas(mesh)

wavenumber = 1.0
operator = Maxwell3D.singlelayer(; wavenumber)
excitation = Maxwell3D.planewave(;
    direction=ẑ, polarization=x̂, wavenumber
)
rhs = assemble((n × excitation) × n, space)

matrix = CFMM.assemble(operator, space)
current, stats = Krylov.gmres(
    matrix, rhs; rtol=1.0e-4, itmax=200, history=true, verbose=1
)
```

The complete runnable example, including the plots, is in
`examples/efie.jl`.

## MFIE

For the magnetic-field integral equation, the nonlocal double-layer operator
uses CFMM while the local ``\frac{1}{2}I`` contribution is assembled directly.
Buffa--Christiansen functions provide the test space:

```julia
mesh = meshsphere(1.0, 0.4)
trialspace = raviartthomas(mesh)
testspace = buffachristiansen(mesh)

permittivity = 1.0
permeability = 1.0
frequency = 1.0
wavenumber = frequency * √(permittivity * permeability)

operator = Maxwell3D.doublelayer(; wavenumber)
excitation = Maxwell3D.planewave(;
    direction=ẑ, polarization=x̂, wavenumber
)
magneticfield = -1 / (im * permeability * frequency) * curl(excitation)
rhs = assemble((n × magneticfield) × n, testspace)

doublelayer = CFMM.assemble(operator, testspace, trialspace)
identity = assemble(NCross(), testspace, trialspace)
matrix = doublelayer + 0.5 * identity
current, stats = Krylov.gmres(
    matrix, rhs; rtol=1.0e-4, itmax=200, history=true, verbose=1
)
```

The complete runnable example, including the plots, is in
`examples/mfie.jl`.
