
struct CFMMMatrixHH3DDoubleLayerTransposed{
    K,OperatorType,FMMType,SparseMatrixType,NormalsType
} <: LinearMaps.LinearMap{K}
    op::OperatorType
    fmm::FMMType
    Btest::SparseMatrixType
    Btrial::SparseMatrixType
    normals::NormalsType

    function CFMMMatrixHH3DDoubleLayerTransposed{K}(
        op, fmm, Btest, Btrial, normals
    ) where {K}
        return new{scalartype(op),typeof(op),typeof(fmm),typeof(Btest),typeof(normals)}(
            op, fmm, Btest, Btrial, normals
        )
    end
end

function Base.size(A::CFMMMatrixHH3DDoubleLayerTransposed, dim=nothing)
    if dim === nothing
        return (size(A.Btest, 1), size(A.Btrial, 2))
    else
        return (dim == 1 ? size(A.Btest, 1) : size(A.Btrial, 2))
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::CFMMMatrixHH3DDoubleLayerTransposed, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    if eltype(x) <: Complex && eltype(A.fmm) <: Real
        y .+= mul!(copy(y), A, real.(x))
        y .+= im .* mul!(copy(y), A, imag.(x))
        return y
    end

    if eltype(x) != eltype(A.fmm)
        xfmm = eltype(A.fmm).(x)
    else
        xfmm = x
    end

    res = A.fmm.A * (A.Btrial * xfmm)
    fmm_res1 = A.normals[:, 1] .* (res)[:, 2]
    fmm_res2 = A.normals[:, 2] .* (res)[:, 3]
    fmm_res3 = A.normals[:, 3] .* (res)[:, 4]

    y .= A.op.alpha .* (A.Btest * (fmm_res1 + fmm_res2 + fmm_res3))

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:CFMMMatrixHH3DDoubleLayerTransposed},
    x::AbstractVector,
)
    LinearMaps.check_dim_mul(y, At, x)

    fill!(y, zero(eltype(y)))

    A = At.lmap

    if eltype(x) <: Complex && eltype(A.fmm) <: Real
        y .+= mul!(copy(y), At, real.(x))
        y .+= im .* mul!(copy(y), At, imag.(x))
        return y
    end

    if eltype(x) != eltype(A.fmm)
        xfmm = eltype(A.fmm).(x)
    else
        xfmm = x
    end

    xx = transpose(A.Btest) * xfmm
    fmm_res1 = (transpose(A.fmm) * (A.normals[:, 1] .* xx))[:, 2]
    fmm_res2 = (transpose(A.fmm) * (A.normals[:, 2] .* xx))[:, 3]
    fmm_res3 = (transpose(A.fmm) * (A.normals[:, 3] .* xx))[:, 4]
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)

    y .= A.op.alpha .* (transpose(A.Btrial) * fmm_res)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:CFMMMatrixHH3DDoubleLayerTransposed},
    x::AbstractVector,
)
    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end
