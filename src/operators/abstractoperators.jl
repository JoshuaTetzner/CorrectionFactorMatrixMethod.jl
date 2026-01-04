using LinearAlgebra
using LinearMaps

include("HH3Dsinglelayer.jl")
include("HH3Ddoublelayer.jl")
include("HH3Ddoublelayertransposed.jl")
include("HH3Dhypersingular.jl")
include("MW3Dsinglelayer.jl")
include("MW3Ddoublelayer.jl")

abstract type FMMFunctor end

## ExaFMMt options interface
struct ExaFMMtFunctor <: FMMFunctor
    p::Int
    ncrit::Int
end

FMMFunctor(; p=8, ncrit=50) = ExaFMMtFunctor(p, ncrit)

function (functor::ExaFMMtFunctor)(operator)
    return error("This function has to be implemented for ExaFMMtFunctor")
end

# FMM wrapper type to hold FMM and if computed its transpose/adjoint
struct FMM{K,FMMType,TransposeFMMType} <: LinearMaps.LinearMap{K}
    A::FMMType
    Aᵀ::TransposeFMMType

    function FMM{K}(A, Aᵀ) where {K}
        return new{K,typeof(A),typeof(Aᵀ)}(A, Aᵀ)
    end
end
Base.eltype(fmm::FMM) = eltype(fmm.A)
Base.size(fmm::FMM, dim=nothing) = size(fmm.A, dim)

LinearMaps._unsafe_mul!(y::AbstractVecOrMat, fmm::FMM, x::AbstractVector) =
    mul!(y, fmm.A, x)
LinearMaps._unsafe_mul!(
    y::AbstractVecOrMat, fmm::LinearMaps.TransposeMap{<:Any,<:FMM}, x::AbstractVector
) = mul!(y, fmm.lmap.Aᵀ, x)
LinearMaps._unsafe_mul!(
    y::AbstractVecOrMat, fmm::LinearMaps.AdjointMap{<:Any,<:FMM}, x::AbstractVector
) = error("Adjoint multiplication not implemented for FMM")
struct SymmetricFMM{K,FMMType} <: LinearMaps.LinearMap{K}
    A::FMMType

    function SymmetricFMM{K}(A) where {K}
        return new{K,typeof(A)}(A)
    end
end
Base.eltype(fmm::SymmetricFMM) = eltype(fmm.A)
Base.size(fmm::SymmetricFMM, dim=nothing) = size(fmm.A, dim)

LinearMaps._unsafe_mul!(y::AbstractVecOrMat, fmm::SymmetricFMM, x::AbstractVector) =
    mul!(y, fmm.A, x)
LinearMaps._unsafe_mul!(
    y::AbstractVecOrMat,
    fmm::LinearMaps.TransposeMap{<:Any,<:SymmetricFMM},
    x::AbstractVector,
) = mul!(y, fmm.lmap.A, x)
LinearMaps._unsafe_mul!(
    y::AbstractVecOrMat, fmm::LinearMaps.AdjointMap{<:Any,<:SymmetricFMM}, x::AbstractVector
) = error("Adjoint multiplication not implemented for SymmetricFMM")

function FMM(A::T) where {T}
    return SymmetricFMM{eltype(A)}(A)
end

function FMM(A::T, Aᵀ::S) where {T,S}
    @assert eltype(A) == eltype(Aᵀ) "FMM and its transpose must have the same element type"
    return FMM{eltype(A)}(A, Aᵀ)
end

# Default alpha and beta functions for operators.
alpha(operator) = 1.0
beta(operator) = 1.0

# Function that returns the positions of the quadrature points on the mesh.
function sources(space, quadorder::Int; args...)
    return error("This functions has to be implemented for ", typeof(space))
end

function potentials(qp::Matrix, X; args...)
    return error("This functions has to be implemented for ", typeof(X))
end

function curlpotentials(qp::Matrix, X; args...)
    return error("This functions has to be implemented for ", typeof(X))
end

function divpotentials(qp::Matrix, X; args...)
    return error("This functions has to be implemented for ", typeof(X))
end

function normals(space; args...)
    return error("This functions has to be implemented for ", typeof(space))
end

# Setup function for FMM given source and target points and fmm options.
function setup(spoints, tpoints, options)
    return error("This function has to be implemented for ", typeof(options))
end

function setup_fmm(
    spoints::Matrix{F}, tpoints::Matrix{F}, options; computetransposeadjoint=false
) where {F<:Real}
    A = setup(spoints, tpoints, options)
    computetransposeadjoint && return FMM(A, setup(tpoints, spoints, options))
    return FMM(A)
end

function (fmmfunctor::FMMFunctor)(
    operator,
    testspace,
    trialspace;
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    computetransposeadjoint=false,
    ntasks=Threads.nthreads(),
)
    testsrcs, testqp = sources(testspace, farquadstrat.outer_rule)
    trialsrcs, trialqp = sources(trialspace, farquadstrat.inner_rule)

    fmm = setup_fmm(
        testsrcs,
        trialsrcs,
        fmmfunctor(operator);
        computetransposeadjoint=computetransposeadjoint,
    )

    return fmmfunctor(operator, testspace, trialspace, testqp, trialqp, fmm; ntasks=ntasks)
end
