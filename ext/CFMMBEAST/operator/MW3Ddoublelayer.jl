
function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    ntasks=Threads.nthreads(),
) where {T<:BEAST.MWDoubleLayer3D}
    Btest, Btrial = MW3DDLpotentialmatrix(testspace, trialspace, testqp, trialqp)

    return CFMMMatrixMW3DDoubleLayer{scalartype(operator)}(operator, fmm, Btest, Btrial)
end

function MW3DDLpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    rc, vals = potentials(testqp, testspace)
    Btest = [
        dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 1])),
        dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 2])),
        dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 3])),
    ]

    testspace == trialspace && return Btest, sparse.(transpose.(Btest))
    rc, vals = potentials(trialqp, trialspace)
    Btrial = [
        dropzeros(sparse(rc[:, 2], rc[:, 1], vals[:, 1])),
        dropzeros(sparse(rc[:, 2], rc[:, 1], vals[:, 2])),
        dropzeros(sparse(rc[:, 2], rc[:, 1], vals[:, 3])),
    ]

    return Btest, Btrial
end
