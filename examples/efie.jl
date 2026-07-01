using BEAST
using CompScienceMeshes
using CorrectionFactorMatrixMethod
using ExaFMMt
using Krylov
using LinearAlgebra
using PlotlyJS

include("plotresults.jl")

mesh = meshsphere(1.0, 0.4)
space = raviartthomas(mesh)

wavenumber = 1.0
operator = Maxwell3D.singlelayer(; wavenumber)
excitation = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber)
rhs = assemble((n × excitation) × n, space)

matrix = CFMM.assemble(operator, space)
current, stats = Krylov.gmres(matrix, rhs; rtol=1.0e-4, itmax=200, history=true, verbose=1)

@show stats
@show norm(matrix * current - rhs) / norm(rhs)

outdir = get(ENV, "CFMM_OUTPUT_DIR", @__DIR__)
plotresults(current, space, excitation, wavenumber, joinpath(outdir, "efie_results.html"))
