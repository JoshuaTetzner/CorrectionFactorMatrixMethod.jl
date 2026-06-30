struct PetrovGalerkinCFMM{K,CorrectedNearsType,FMMType,SchedulerType} <:
       LinearMaps.LinearMap{K}
    correctednears::CorrectedNearsType
    fmm::FMMType
    dim::Tuple{Int,Int}
    scheduler::SchedulerType

    function PetrovGalerkinCFMM{K}(correctednears, fmm, dim, scheduler) where {K}
        return new{K,typeof(correctednears),typeof(fmm),typeof(scheduler)}(
            correctednears, fmm, dim, scheduler
        )
    end
end

function Base.size(A::PetrovGalerkinCFMM, dim=nothing)
    return dim === nothing ? A.dim : A.dim[dim]
end

function scalartype(operator) end

"""
    PetrovGalerkinCFMM(operator, testspace, trialspace, tree; kwargs...)

Construct a matrix-free Petrov-Galerkin operator whose far interactions are
evaluated by a fast multipole method and whose near interactions are corrected
by direct quadrature.

The concrete operator and space methods are supplied by package extensions.
Loading `BEAST`, `CompScienceMeshes`, and `ExaFMMt` enables the supported
boundary-element implementation.
"""
function PetrovGalerkinCFMM(
    operator,
    testspace,
    trialspace,
    tree;
    fmmfunctor=FMMFunctor(),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    scheduler=DynamicScheduler(),
    isnear=H2Trees.isnear,
    computetransposeadjoint=false,
)
    correctednears = assemblecorrectednears(
        operator,
        testspace,
        trialspace,
        tree;
        nearquadstrat=nearquadstrat,
        farquadstrat=farquadstrat,
        scheduler=scheduler,
        isnear=isnear,
    )

    fmm = fmmfunctor(
        operator,
        testspace,
        trialspace;
        scheduler=scheduler,
        farquadstrat=farquadstrat,
        computetransposeadjoint=computetransposeadjoint,
    )

    return PetrovGalerkinCFMM{scalartype(operator)}(
        correctednears, fmm, (length(testspace), length(trialspace)), scheduler
    )
end

function LinearMaps.mul!(y::AbstractVector, A::PetrovGalerkinCFMM, x::AbstractVector)
    LinearMaps.mul!(y, A.fmm, x)
    mul!(y, A.correctednears, x, true, true)
    return y
end

function LinearMaps.mul!(
    y::AbstractVector,
    A::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinCFMM},
    x::AbstractVector,
)
    LinearMaps.mul!(y, transpose(A.lmap.fmm), x)
    mul!(y, transpose(A.lmap.correctednears), x, true, true)
    return y
end

function LinearMaps.mul!(
    y::AbstractVector,
    A::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinCFMM},
    x::AbstractVector,
)
    LinearMaps.mul!(y, adjoint(A.lmap.fmm), x)
    mul!(y, adjoint(A.lmap.correctednears), x, true, true)
    return y
end
