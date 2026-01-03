
function CorrectionFactorMatrixMethod.AbstractCorrectedKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    nearquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    farquadstrat=SafeDoubleNumQStrat(operator, testspace, trialspace),
)
    return CorrectionFactorMatrixMethod.BEASTCorrectedKernelMatrix{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=nearquadstrat),
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=farquadstrat),
    )
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    @views f.matrix[m, n] += v
    return nothing
end

struct BlockCorrectionFunctor{M}
    matrix::M
end

function (f::BlockCorrectionFunctor)(v, m, n)
    f.matrix[m, n] -= v
    return nothing
end

function (blk::CorrectionFactorMatrixMethod.BEASTCorrectedKernelMatrix)(
    tdata, sdata, matrixblock
)
    blk.nearassembler(tdata, sdata, BlockStoreFunctor(matrixblock))
    blk.correctionassembler(tdata, sdata, BlockCorrectionFunctor(matrixblock))
    return nothing
end
