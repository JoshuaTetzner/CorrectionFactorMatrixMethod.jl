function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    ntasks=Threads.nthreads(),
) where {T<:BEAST.HH3DDoubleLayerTransposedFDBIO}
    Btest, Btrial, normals = HH3DDLTpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CFMMMatrixHH3DDoubleLayerTransposed{scalartype(operator)}(
        operator, fmm, Btest, Btrial, normals
    )
end

function HH3DDLTpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    normals = normals(testqp)
    rc, vals = potentials(testqp, testspace)
    Btest = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))
    testspace == trialspace && return Btest, sparse(transpose(Btest)), normals

    rc_trial, vals_trial = potentials(trialqp, trialspace)
    Btrial = dropzeros(sparse(rc_trial[:, 2], rc_trial[:, 1], vals_trial))
    return Btest, Btrial, normals
end
