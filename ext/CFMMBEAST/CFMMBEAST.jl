module CFMMBEAST

using CorrectionFactorMatrixMethod
import CorrectionFactorMatrixMethod: FMMFunctor, potentials, sources

using BEAST
using CompScienceMeshes
using SparseArrays

include("kernelmatrix.jl")
include("quadstrat.jl")
include("collocation.jl")
include("operator/HH3Dsinglelayer.jl")

function CorrectionFactorMatrixMethod.defaultnearquadstrat(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

function CorrectionFactorMatrixMethod.scalartype(operator::BEAST.IntegralOperator)
    return BEAST.scalartype(operator)
end

function CorrectionFactorMatrixMethod.alpha(operator::BEAST.IntegralOperator)
    return operator.alpha
end

function CorrectionFactorMatrixMethod.beta(operator::BEAST.IntegralOperator)
    return operator.beta
end

end
