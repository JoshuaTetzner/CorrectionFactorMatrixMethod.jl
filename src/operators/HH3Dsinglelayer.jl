using LinearAlgebra
using LinearMaps

struct CFMMMatrixHH3DSingleLayer{K,OperatorType,FMMType,SparseMatrixType} <:
       LinearMaps.LinearMap{K}
    operator::OperatorType
    fmm::FMMType
    Bₜ::SparseMatrixType
    Bₛ::SparseMatrixType

    function CFMMMatrixHH3DSingleLayer{K}(operator, fmm, Bₜ, Bₛ) where {K}
        return new{scalartype(operator),typeof(operator),typeof(fmm),typeof(Bₜ)}(
            operator, fmm, Bₜ, Bₛ
        )
    end
end

function Base.size(A::CFMMMatrixHH3DSingleLayer, dim=nothing)
    if dim === nothing
        return (size(A.Bₜ, 1), size(A.Bₛ, 2))
    else
        return (dim == 1 ? size(A.Bₜ, 1) : size(A.Bₛ, 2))
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::CFMMMatrixHH3DSingleLayer, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(eltype(x)))

    if eltype(x) <: Complex && eltype(A.fmm) <: Real
        y .+= mul!(copy(y), A, real.(x))
        y .+= im .* mul!(copy(y), A, imag.(x))
        return y
    end

    eltype(x) != eltype(A.fmm) ? (xfmm = eltype(A.fmm).(x)) : (xfmm = x)

    y .= alpha(A.operator) .* (A.Bₜ * (A.fmm.A * (A.Bₛ * xfmm))[:, 1])

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:CFMMMatrixHH3DSingleLayer},
    x::AbstractVector,
)
    LinearMaps.check_dim_mul(y, At, x)
    fill!(y, zero(eltype(x)))
    A = At.lmap

    if eltype(x) <: Complex && eltype(A.fmm) <: Real
        y .+= mul!(copy(y), At, real.(x))
        y .+= im .* mul!(copy(y), At, imag.(x))
        return y
    end

    eltype(x) != eltype(A.fmm) ? (xfmm = eltype(A.fmm).(x)) : (xfmm = x)

    y .=
        A.op.alpha .*
        (transpose(A.Bₛ) * (transpose(A.fmm) * (transpose(A.Bₜ) * xfmm))[:, 1])

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:CFMMMatrixHH3DSingleLayer},
    x::AbstractVector,
)
    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end
