function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    scheduler,
) where {T<:BEAST.HH3DDoubleLayerFDBIO}
    Btest, Btrial, normals = HH3DDLpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CorrectionFactorMatrixMethod.CFMMMatrixHH3DDoubleLayer{scalartype(operator)}(
        operator, fmm, Btest, Btrial, normals
    )
end

function HH3DDLpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    trialnormals = normals(trialqp)
    rc, vals = potentials(testqp, testspace)
    Btest = dropzeros(sparse(rc[:, 2], rc[:, 1], vals))

    testspace == trialspace && return Btest, sparse(transpose(Btest)), trialnormals
    rc, vals = potentials(trialqp, trialspace)
    Btrial = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))

    return Btest, Btrial, trialnormals
end
