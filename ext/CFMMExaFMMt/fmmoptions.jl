function exafmmoptions(
    gamma::T, fmm::CorrectionFactorMatrixMethod.ExaFMMtFunctor
) where {T<:Val{0}}
    return LaplaceFMMOptions(; p=fmm.p, ncrit=fmm.ncrit)
end
function exafmmoptions(
    gamma::T, fmm::CorrectionFactorMatrixMethod.ExaFMMtFunctor
) where {T<:Real}
    iszero(gamma) && return LaplaceFMMOptions(; p=fmm.p, ncrit=fmm.ncrit)
    return ModifiedHelmholtzFMMOptions(gamma; p=fmm.p, ncrit=fmm.ncrit)
end
function exafmmoptions(
    gamma::T, fmm::CorrectionFactorMatrixMethod.ExaFMMtFunctor
) where {T<:Complex}
    iszero(gamma) && return LaplaceFMMOptions(; p=fmm.p, ncrit=fmm.ncrit)
    return HelmholtzFMMOptions(-gamma / im; p=fmm.p, ncrit=fmm.ncrit)
end

function (functor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::BEAST.IntegralOperator
)
    return exafmmoptions(operator.gamma, functor)
end
