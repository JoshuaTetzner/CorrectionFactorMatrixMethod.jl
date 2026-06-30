"""
    CorrectionFactorMatrixMethod

Matrix-free boundary-element operators that combine a fast multipole
approximation of the far interactions with directly assembled near-field
corrections.

The far field is evaluated by a fast multipole method, while the near
interactions are assembled with the boundary-element quadrature and corrected
for the part already represented by the FMM:

    A x ≈ A_fmm x + (A_near - A_fmm,near) x

The generic correction and linear-map machinery lives in the core package;
support for concrete operators and spaces is provided through package
extensions for [BEAST](https://github.com/krcools/BEAST.jl),
[CompScienceMeshes](https://github.com/krcools/CompScienceMeshes.jl), and
[ExaFMMt](https://github.com/JoshuaTetzner/ExaFMMt.jl).

The main entry point is [`CFMM.assemble`](@ref); see also
[`PetrovGalerkinCFMM`](@ref) and [`FMMFunctor`](@ref).
"""
module CorrectionFactorMatrixMethod

using BlockSparseMatrices
using H2Trees
using LinearMaps
using LinearAlgebra
using MKL
using OhMyThreads
using SparseArrays

# The quadrature is handled in the BEM library (e.g., BEAST), so we just
# define some placeholder types here for the quadpoints used in the FMM.
# The selfinteractions are not handled by the FMM and have to be 0 for this quadtsrat.
struct SafeDoubleNumQStrat{R}
    outer_rule::R
    inner_rule::R
end

function (qs::SafeDoubleNumQStrat)(op, testspace, trialspace)
    return qs
end

struct SafeDoubleQuadRule{P,Q}
    outer_quad_points::P
    inner_quad_points::Q
end

function defaultfarquadstrat(op, testspace, trialspace)
    return SafeDoubleNumQStrat(3, 3)
end

# has to be defiend for the operator and space combinations available.
function defaultnearquadstrat(op, testspace, trialspace)
    return error("This function has to be implemented for these types.")
end

include("kernelmatrix/abstractcorrectedkernelmatrix.jl")
include("kernelmatrix/beastcorrectedkernelmatrix.jl")

include("nearinteractions.jl")

include("operators/abstractoperators.jl")

include("CFMM/petrovgalerkincfmm.jl")
include("CFMM/assemble.jl")

export CFMM, PetrovGalerkinCFMM

end
