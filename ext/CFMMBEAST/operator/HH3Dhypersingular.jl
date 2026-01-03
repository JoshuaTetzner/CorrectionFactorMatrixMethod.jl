using BEAST
using ExaFMMt
using LinearAlgebra
using LinearMaps
using SparseArrays

function FMMMatrix(
    ::Type{BEAST.HH3DHyperSingularFDBIO},
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
    fmm::FMM,
)
    normals_trial, B1curl, B2curl, B3curl, B, normals_test, B1curl_test, B2curl_test, B3curl_test, B_test = sample_basisfunctions(
        op, test_functions, trial_functions, testqp, trialqp
    )

    return FMMMatrixHS(
        fmm,
        fmm_t,
        op,
        normals_trial,
        normals_test,
        B1curl,
        B2curl,
        B3curl,
        B1curl_test,
        B2curl_test,
        B3curl_test,
        B,
        B_test,
        BtCB,
        fullmat,
        size(fullmat)[1],
        size(fullmat)[2],
    )
end

function sample_basisfunctions(
    op::BEAST.HH3DHyperSingularFDBIO,
    test_functions::BEAST.Space,
    trial_functions::BEAST.Space,
    testqp::Matrix,
    trialqp::Matrix,
)
    normals_trial = getnormals(trialqp)
    normals_test = getnormals(testqp)
    rc_curl, vals_curl = sample_curlbasisfunctions(trialqp, trial_functions)
    B1curl = dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 1]))
    B2curl = dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 2]))
    B3curl = dropzeros(sparse(rc_curl[:, 1], rc_curl[:, 2], vals_curl[:, 3]))
    B1curl_test, B2curl_test, B3curl_test = B1curl, B2curl, B3curl

    rc, vals = sample_basisfunctions(op, trialqp, trial_functions)
    B = dropzeros(sparse(rc[:, 1], rc[:, 2], vals))
    B_test = B

    if test_functions != trial_functions
        normals_test = getnormals(testqp)
        rc_curl, vals_curl = sample_curlbasisfunctions(testqp, test_functions)
        B1curl_test = dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 1]))
        B2curl_test = dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 2]))
        B3curl_test = dropzeros(sparse(rc_curl[:, 2], rc_curl[:, 1], vals_curl[:, 3]))

        rc, vals = sample_basisfunctions(op, testqp, test_functions)
        B_test = dropzeros(sparse(rc[:, 2], rc[:, 1], vals))
    else
        B1curl_test = sparse(transpose(B1curl))
        B2curl_test = sparse(transpose(B2curl))
        B3curl_test = sparse(transpose(B3curl))
        B_test = sparse(transpose(B))
    end

    return normals_trial,
    B1curl, B2curl, B3curl, B, normals_test, B1curl_test, B2curl_test, B3curl_test,
    B_test
end
