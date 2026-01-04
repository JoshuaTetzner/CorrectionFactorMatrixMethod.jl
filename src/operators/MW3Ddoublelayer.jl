
struct CFMMMatrixMW3DDoubleLayer{K,OperatorType,FMMType,SparseMatrixType} <:
       LinearMaps.LinearMap{K}
    operator::OperatorType
    fmm::FMMType
    Btest::Vector{SparseMatrixType}
    Btrial::Vector{SparseMatrixType}

    function CFMMMatrixMW3DDoubleLayer{K}(operator, fmm, Btest, Btrial) where {K}
        return new{scalartype(operator),typeof(operator),typeof(fmm),typeof(Btest)}(
            operator, fmm, Btest, Btrial
        )
    end
end

function Base.size(fmat::CFMMMatrixMW3DDoubleLayer, dim=nothing)
    if dim === nothing
        return (size(fmat.Btest[1], 1), size(fmat.Btrial[1], 2))
    else
        return (dim == 1 ? size(fmat.Btest[1], 1) : size(fmat.Btrial[1], 2))
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::CFMMMatrixMW3DDoubleLayer{K}, x::AbstractVector
) where {K}
    LinearMaps.check_dim_mul(y, A, x)

    if eltype(x) != K
        x = K.(x)
    end
    fill!(y, zero(K))

    res1 = (A.fmm * (A.Btrial[1] * x))[:, 2:4]
    res2 = (A.fmm * (A.Btrial[2] * x))[:, 2:4]
    res3 = (A.fmm * (A.Btrial[3] * x))[:, 2:4]

    y1 = A.Btest[1] * (res3[:, 2] - res2[:, 3])
    y2 = A.Btest[2] * (res1[:, 3] - res3[:, 1])
    y3 = A.Btest[3] * (res2[:, 1] - res1[:, 2])

    y .= (y1 + y2 + y3)

    return A.operator.alpha .* y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:CFMMMatrixMW3DDoubleLayer{K}},
    x::AbstractVector,
) where {K}
    LinearMaps.check_dim_mul(y, At, x)

    A = At.lmap
    if eltype(x) != K
        x = K.(x)
    end
    fill!(y, zero(K))

    res1 = (transpose(A.fmm) * (transpose(A.Btest[1]) * x))[:, 2:4]
    res2 = (transpose(A.fmm) * (transpose(A.Btest[2]) * x))[:, 2:4]
    res3 = (transpose(A.fmm) * (transpose(A.Btest[3]) * x))[:, 2:4]

    y1 = transpose(A.Btrial[1]) * (res3[:, 2] - res2[:, 3])
    y2 = transpose(A.Btrial[2]) * (res1[:, 3] - res3[:, 1])
    y3 = transpose(A.Btrial[3]) * (res2[:, 1] - res1[:, 2])

    y .= (y1 + y2 + y3)

    return A.operator.alpha .* y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:CFMMMatrixMW3DDoubleLayer},
    x::AbstractVector,
)
    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end
