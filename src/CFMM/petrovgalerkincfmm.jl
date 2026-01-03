struct PetrovGalerkinCFMM{K,CorrectedNearsType,FMMType} <: LinearMaps.LinearMap{K}
    correctednears::CorrectedNearsType
    fmm::FMMType
    dim::Tuple{Int,Int}
    ntasks::Int

    function PetrovGalerkinCFMM{K}(correctednears, fmm, dim, ntasks) where {K}
        return new{K,typeof(correctednears),typeof(fmm)}(correctednears, fmm, dim, ntasks)
    end
end

function Base.size(A::PetrovGalerkinCFMM, dim=nothing)
    return dim === nothing ? A.dim : A.dim[dim]
end

function scalartype(operator) end

function PetrovGalerkinCFMM(
    operator,
    testspace,
    trialspace,
    tree;
    fmmfunctor=FMMFunctor(),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    ntasks=Threads.nthreads(),
    isnear=H2Trees.isnear,
    computetransposeadjoint=false,
)
    correctedkernelmatrix = AbstractCorrectedKernelMatrix(
        operator,
        testspace,
        trialspace;
        nearquadstrat=nearquadstrat,
        farquadstrat=farquadstrat,
    )
    values, nearvalues = nearinteractions(tree; isnear=isnear)
    blocks = tmap(values, nearvalues) do v, nv
        blk = zeros(scalartype(operator), length(v), length(nv))
        correctedkernelmatrix(v, nv, blk)
        return blk
    end
    correctednears = BlockSparseMatrix(
        blocks, values, nearvalues, length(testspace), length(trialspace)
    )

    fmm = fmmfunctor(
        operator,
        testspace,
        trialspace;
        ntasks=ntasks,
        farquadstrat=farquadstrat,
        computetransposeadjoint=computetransposeadjoint,
    )

    return PetrovGalerkinCFMM{scalartype(operator)}(
        correctednears, fmm, (length(testspace), length(trialspace)), ntasks
    )
end

function LinearMaps.mul!(
    y::AbstractVector{K}, A::PetrovGalerkinCFMM{K}, x::AbstractVector{K}
) where {K}
    LinearMaps.mul!(y, A.fmm, x)
    y += A.correctednears * x
    return y
end

function LinearMaps.mul!(
    y::AbstractVector{K},
    A::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinCFMM{K}},
    x::AbstractVector{K},
) where {K}
    LinearMaps.mul!(y, transpose(A.lmap.fmm), x)
    y += transpose(A.lmap.correctednears) * x
    return y
end

function LinearMaps.mul!(
    y::AbstractVector{K},
    A::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinCFMM{K}},
    x::AbstractVector{K},
) where {K}
    LinearMaps.mul!(y, adjoint(A.lmap.fmm), x)
    y += adjoint(A.lmap.correctednears) * x
    return y
end
