function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    ntasks=Threads.nthreads(),
) where {T<:BEAST.HH3DDoubleLayerTransposedFDBIO}
    Bₜ, Bₛ, normals = HH3DDLTpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CFMMMatrixHH3DDoubleLayerTransposed{scalartype(operator)}(
        operator, fmm, Bₜ, Bₛ, normals
    )
end

function HH3DDLTpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    normals = getnormals(testqp)
    rc, vals = potentials(trialqp, trialspace)
    Bₛ = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))
    testspace == trialspace && return sparse(transpose(Bₛ)), Bₛ

    rc_test, vals_test = potentials(testqp, testspace)
    Bₜ = dropzeros(sparse(rc_test[:, 2], rc_test[:, 1], vals_test))
    return Bₜ, Bₛ, normals
end
