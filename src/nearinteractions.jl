function assemblecorrectednears(
    operator,
    testspace,
    trialspace,
    tree;
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
    isnear=H2Trees.isnear,
)
    matrix = AbstractCorrectedKernelMatrix(
        operator,
        testspace,
        trialspace;
        nearquadstrat=nearquadstrat,
        farquadstrat=farquadstrat,
    )
    values, nearvalues = H2Trees.nearinteractions(tree; isnear=isnear)
    blocks = tmap(values, nearvalues; scheduler=scheduler) do v, nv
        block = zeros(scalartype(operator), length(v), length(nv))
        matrix(block, v, nv)
        return block
    end
    scheduler = isempty(blocks) ? SerialScheduler() : scheduler
    return BlockSparseMatrix(
        blocks,
        values,
        nearvalues,
        length(testspace),
        length(trialspace);
        scheduler=scheduler,
    )
end
