function (fmmfunctor::CorrectionFactorMatrixMethod.ExaFMMtFunctor)(
    operator::T,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm;
    ntasks=Threads.nthreads(),
) where {T<:BEAST.HH3DHyperSingularFDBIO}
    testnormals, curlBtest, Btest, trialnormals, curlBtrial, Btrial = HH3DHSpotentialmatrix(
        testspace, trialspace, testqp, trialqp
    )

    return FMMMatrixHS{scalartype(operator)}(
        operator, fmm, testnormals, trialnormals, curlBtest, curlBtrial, Btest, Btrial
    )
end

function HH3DHSpotentialmatrix(
    testspace::BEAST.Space, trialspace::BEAST.Space, testqp::Matrix, trialqp::Matrix
)
    testnormals = getnormals(testqp)
    rc_curl, vals_curl = curlpotentials(testqp, testspace)
    curlBtest = [
        dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 1])),
        dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 2])),
        dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 3])),
    ]

    rc, vals = potentials(testqp, testspace)
    Btest = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))

    testspace == trialspace && return testnormals,
    curlBtest, Btest, testnormals, sparse.(transpose(curlBtest)),
    sparse(transpose(Btest))

    trialnormals = getnormals(trialqp)
    rc_curl, vals_curl = curlpotentials(trialqp, trialspace)
    curlBtrial = [
        dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 1])),
        dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 2])),
        dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 3])),
    ]

    rc, vals = potentials(trialqp, trialspace)
    Btrial = dropzeros(sparse(rc[:, 2], rc[:, 1], vals))

    return testnormals, curlBtest, Btest, trialnormals, curlBtrial, Btrial
end
