module CFMMBEAST

using CorrectionFactorMatrixMethod
import CorrectionFactorMatrixMethod: curlpotentials, divpotentials, normals, potentials

using BEAST
using CompScienceMeshes
using H2Trees
using SparseArrays

include("kernelmatrix.jl")
include("quadstrat.jl")
include("collocation.jl")
include("operator/HH3Dsinglelayer.jl")
include("operator/HH3Ddoublelayer.jl")
include("operator/HH3Ddoublelayertransposed.jl")
include("operator/HH3Dhypersingular.jl")
include("operator/MW3Dsinglelayer.jl")
include("operator/MW3Ddoublelayer.jl")

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

function CorrectionFactorMatrixMethod.defaultminhalfsize(
    testspace::BEAST.Space, trialspace::BEAST.Space
)
    _, testhalfsize = H2Trees.boundingbox(BEAST.positions(testspace))
    _, trialhalfsize = H2Trees.boundingbox(BEAST.positions(trialspace))
    return max(testhalfsize, trialhalfsize) / 2^10
end

end
