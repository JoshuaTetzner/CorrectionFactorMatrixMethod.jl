
struct CFMMMatrixMW3DSingleLayer{K,OperatorType,FMMType,SparseMatrixType} <:
       LinearMaps.LinearMap{K}
    operator::OperatorType
    fmm::FMMType
    Btest::Vector{SparseMatrixType}
    Btrial::Vector{SparseMatrixType}
    divBtest::SparseMatrixType
    divBtrial::SparseMatrixType

    function CFMMMatrixMW3DSingleLayer{K}(
        operator, fmm, Btest, Btrial, divBtest, divBtrial
    ) where {K}
        return new{scalartype(operator),typeof(operator),typeof(fmm),typeof(Btest)}(
            operator, fmm, Btest, Btrial, divBtest, divBtrial
        )
    end
end

function Base.size(fmat::CFMMMatrixMW3DSingleLayer, dim=nothing)
    if dim === nothing
        return (size(fmat.Btest[1], 1), size(fmat.Btrial[1], 2))
    else
        return (dim == 1 ? size(fmat.Btest[1], 1) : size(fmat.Btrial[1], 2))
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::CFMMMatrixMW3DSingleLayer{K}, x::AbstractVector
) where {K}
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != K
        x = K.(x)
    end
    fill!(y, zero(K))

    res1 = A.Btest[1] * (A.fmm * (A.Btrial[1] * x))[:, 1]
    res2 = A.Btest[2] * (A.fmm * (A.Btrial[2] * x))[:, 1]
    res3 = A.Btest[3] * (A.fmm * (A.Btrial[3] * x))[:, 1]

    y1 = (A.operator.α .* (res1 + res2 + res3))
    y2 = -(A.operator.β) .* (A.divBtest * (A.fmm * (A.divBtrial * x))[:, 1])

    y .= (y1 - y2)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:CFMMMatrixMW3DSingleLayer{K}},
    x::AbstractVector,
) where {K}
    LinearMaps.check_dim_mul(y, At, x)

    A = At.lmap
    if eltype(x) != K
        x = K.(x)
    end
    fill!(y, zero(K))

    res1 = transpose(A.Btrial[1]) * (transpose(A.fmm) * (transpose(A.Btest[1]) * x))[:, 1]
    res2 = transpose(A.Btrial[2]) * (transpose(A.fmm) * (transpose(A.Btest[2]) * x))[:, 1]
    res3 = transpose(A.Btrial[3]) * (transpose(A.fmm) * (transpose(A.Btest[3]) * x))[:, 1]

    y1 = (A.operator.α .* (res1 + res2 + res3))
    y2 =
        -(A.operator.β) .*
        (transpose(A.divBtrial) * (transpose(A.fmm) * (transpose(A.divBtest) * x))[:, 1])

    y .= (y1 - y2)

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:CFMMMatrixMW3DSingleLayer},
    x::AbstractVector,
)
    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end
