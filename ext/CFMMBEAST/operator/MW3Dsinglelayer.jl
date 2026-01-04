
function (::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    ntasks=Threads.nthreads(),
) where {T<:BEAST.MWSingleLayer3D}
    Btest, divBtest, Btrial, divBtrial = MW3DSLpotentialmatrix(
        testspace, trialspace, testqp, trialqp
    )

    return CFMMMatrixMW3DSingleLayer{scalartype(operator)}(
        operator, fmm, Btest, Btrial, divBtest, divBtrial
    )
end

function MW3DSLpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    rc, vals = potentials(testqp, testspace)
    Btest = [
        dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 1])),
        dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 2])),
        dropzeros(sparse(rc[:, 1], rc[:, 2], vals[:, 3])),
    ]

    rcdiv, valsdiv = divpotentials(testqp, testspace)
    divBtest = dropzeros(sparse(rcdiv[:, 1], rcdiv[:, 2], valsdiv))

    testspace == trialspace &&
        return Btest, divBtest, sparse.(transpose.(Btest)), sparse(transpose(divBtest))

    rc, vals = potentials(trialqp, trialspace)
    Btrial = [
        dropzeros(sparse(rc[:, 2], rc[:, 1], vals[:, 1])),
        dropzeros(sparse(rc[:, 2], rc[:, 1], vals[:, 2])),
        dropzeros(sparse(rc[:, 2], rc[:, 1], vals[:, 3])),
    ]
    rcdiv, valsdiv = divpotentials(trialqp, trialspace)
    divBtrial = dropzeros(sparse(rcdiv[:, 2], rcdiv[:, 1], valsdiv))

    return Btest, divBtest, Btrial, divBtrial
end
