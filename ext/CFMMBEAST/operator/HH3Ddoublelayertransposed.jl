function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    scheduler,
) where {T<:BEAST.HH3DDoubleLayerTransposedFDBIO}
    Btest, Btrial, normals = HH3DDLTpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CorrectionFactorMatrixMethod.CFMMMatrixHH3DDoubleLayerTransposed{
        scalartype(operator)
    }(
        operator, fmm, Btest, Btrial, normals
    )
end

function HH3DDLTpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    testnormals = normals(testqp)
    rc, vals = potentials(testqp, testspace)
    Btest = dropzeros(sparse(rc[:, 2], rc[:, 1], vals))
    testspace == trialspace && return Btest, sparse(transpose(Btest)), testnormals

    rc, vals = potentials(trialqp, trialspace)
    Btrial = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))
    return Btest, Btrial, testnormals
end
