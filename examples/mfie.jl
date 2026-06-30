using BEAST
using CompScienceMeshes
using CorrectionFactorMatrixMethod
using ExaFMMt
using Krylov
using LinearAlgebra
using PlotlyJS

include("plotresults.jl")

mesh = meshsphere(1.0, 0.4)
trialspace = raviartthomas(mesh)
testspace = buffachristiansen(mesh)

permittivity = 1.0
permeability = 1.0
frequency = 1.0
wavenumber = frequency * √(permittivity * permeability)

operator = Maxwell3D.doublelayer(; wavenumber)
excitation = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber)
magneticfield = -1 / (im * permeability * frequency) * curl(excitation)
rhs = assemble((n × magneticfield) × n, testspace)

doublelayer = CFMM.assemble(operator, testspace, trialspace)
identity = assemble(NCross(), testspace, trialspace)
matrix = doublelayer + 0.5 * identity
current, stats = Krylov.gmres(matrix, rhs; rtol=1.0e-4, itmax=200, history=true, verbose=1)

@show stats
@show norm(matrix * current - rhs) / norm(rhs)

plotresults(
    current, trialspace, excitation, wavenumber, joinpath(@__DIR__, "mfie_results.html")
)
