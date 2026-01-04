
struct CFMMMatrixHH3DDoubleLayer{K,OperatorType,FMMType,SparseMatrixType,NormalsType} <:
       LinearMaps.LinearMap{K}
    operator::OperatorType
    fmm::FMMType
    Btest::SparseMatrixType
    Btrial::SparseMatrixType
    normals::NormalsType

    function CFMMMatrixHH3DDoubleLayer{K}(operator, fmm, Btest, Btrial, normals) where {K}
        return new{
            scalartype(operator),typeof(operator),typeof(fmm),typeof(Btest),typeof(normals)
        }(
            operator, fmm, Btest, Btrial, normals
        )
    end
end

function Base.size(A::CFMMMatrixHH3DDoubleLayer, dim=nothing)
    if dim === nothing
        return (size(A.Btest, 1), size(A.Btrial, 2))
    else
        return (dim == 1 ? size(A.Btest, 1) : size(A.Btrial, 2))
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::CFMMMatrixHH3DDoubleLayer{K}, x::AbstractVector
) where {K}
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(K))

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

    fmm_res1 = A.Btest * (A.fmm * (A.normals[:, 1] .* (A.Btrial * xfmm)))[:, 2]
    fmm_res2 = A.Btest * (A.fmm * (A.normals[:, 2] .* (A.Btrial * xfmm)))[:, 3]
    fmm_res3 = A.Btest * (A.fmm * (A.normals[:, 3] .* (A.Btrial * xfmm)))[:, 4]
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)
    y .= A.operator.alpha .* fmm_res

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:CFMMMatrixHH3DDoubleLayer},
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

    xx = -1 .* (transpose(A.fmm) * (transpose(A.Btest) * xfmm))
    test = transpose(A.Btrial)

    fmm_res1 = test * (A.normals[:, 1] .* xx[:, 2])
    fmm_res2 = test * (A.normals[:, 2] .* xx[:, 3])
    fmm_res3 = test * (A.normals[:, 3] .* xx[:, 4])
    fmm_res = -(fmm_res1 + fmm_res2 + fmm_res3)
    y .= A.operator.alpha .* fmm_res

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:CFMMMatrixHH3DDoubleLayer},
    x::AbstractVector,
)
    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end
