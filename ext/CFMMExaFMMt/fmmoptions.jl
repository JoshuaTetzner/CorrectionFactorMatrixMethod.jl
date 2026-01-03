exafmmoptions(
    gamma::T, fmm::CorrectionFactorMatrixMethod.ExaFMMtFunctor
) where {T<:Val{0}} = LaplaceFMMOptions(; p=fmm.p, ncrit=fmm.ncrit)
#TODO: Write unit tests for the ModifiedHelmholtzFMMOptions
exafmmoptions(gamma::T, fmm::CorrectionFactorMatrixMethod.ExaFMMtFunctor) where {T<:Real} =
    ModifiedHelmholtzFMMOptions(gamma; p=fmm.p, ncrit=fmm.ncrit)
exafmmoptions(
    gamma::T, fmm::CorrectionFactorMatrixMethod.ExaFMMtFunctor
) where {T<:Complex} = HelmholtzFMMOptions(-gamma / im; p=fmm.p, ncrit=fmm.ncrit)

function (functor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::BEAST.IntegralOperator
)
    return exafmmoptions(operator.gamma, functor)
end
