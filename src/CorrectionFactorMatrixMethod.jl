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

export PetrovGalerkinCFMM

end
