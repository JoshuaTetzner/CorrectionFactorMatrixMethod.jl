function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    scheduler,
) where {T<:BEAST.HH3DSingleLayerFDBIO}
    Btest, Btrial = HH3DSLpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CorrectionFactorMatrixMethod.CFMMMatrixHH3DSingleLayer{scalartype(operator)}(
        operator, fmm, Btest, Btrial
    )
end

function HH3DSLpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    rc, vals = potentials(testqp, testspace)
    Btest = dropzeros(sparse(rc[:, 2], rc[:, 1], vals))

    testspace == trialspace && return Btest, sparse(transpose(Btest))

    rc, vals = potentials(trialqp, trialspace)
    Btrial = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))
    return Btest, Btrial
end
