# Regenerate the pre-rendered plot assets embedded in the documentation.
# Run from the package root:
#
#   julia --startup-file=no docs/render_examples.jl
#
# Commit the generated files in docs/src/assets/examples/ afterwards.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", "examples"))
Pkg.develop(; path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using BEAST
using CompScienceMeshes
using CorrectionFactorMatrixMethod
using ExaFMMt
using Krylov
using LinearAlgebra
using PlotlyJS

outdir = joinpath(@__DIR__, "src", "assets", "examples")
mkpath(outdir)
ENV["CFMM_OUTPUT_DIR"] = outdir

include(joinpath(@__DIR__, "..", "examples", "efie.jl"))
include(joinpath(@__DIR__, "..", "examples", "mfie.jl"))

@info "Plots written to $outdir"
