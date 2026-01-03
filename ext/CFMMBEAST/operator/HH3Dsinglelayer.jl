function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    ntasks=Threads.nthreads(),
) where {T<:BEAST.HH3DSingleLayerFDBIO}
    Bₜ, Bₛ = HH3DSLpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CorrectionFactorMatrixMethod.CFMMMatrixHH3DSingleLayer{scalartype(operator)}(
        operator, fmm, Bₜ, Bₛ
    )
end

function HH3DSLpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    rc, vals = potentials(trialqp, trialspace)
    Bₛ = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))

    testspace == trialspace && return sparse(transpose(Bₛ)), Bₛ

    rc_test, vals_test = potentials(testqp, testspace)
    Bₜ = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test))
    return Bₜ, Bₛ
end
