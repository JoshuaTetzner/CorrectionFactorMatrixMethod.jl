struct CFMMMatrixHH3DHyperSingular{K,OperatorType,FMMType,NormalsType,SparseMatrixType} <:
       LinearMaps.LinearMap{K}
    operator::OperatorType
    fmm::FMMType
    testnormals::NormalsType
    trialnormals::NormalsType
    curlBtest::Vector{SparseMatrixType}
    curlBtrial::Vector{SparseMatrixType}
    Btest::SparseMatrixType
    Btrial::SparseMatrixType

    function CFMMMatrixHH3DHyperSingular{K}(
        operator, fmm, testnormals, trialnormals, curlBtest, curlBtrial, Btest, Btrial
    ) where {K}
        return new{
            scalartype(operator),
            typeof(operator),
            typeof(fmm),
            typeof(testnormals),
            typeof(Btest),
        }(
            op, fmm, testnormals, trialnormals, curlBtest, curlBtrial, Btest, Btrial
        )
    end
end

function Base.size(A::CFMMMatrixHH3DHyperSingular, dim=nothing)
    if dim === nothing
        return (size(A.Btest, 1), size(A.Btrial, 2))
    else
        return (dim == 1 ? size(A.Btest, 1) : size(A.Btrial, 2))
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::CFMMMatrixHH3DHyperSingular, x::AbstractVector
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
    if A.op.alpha != 0.0
        fmm_res1 =
            A.testnormals[:, 1] .*
            (A.fmm * (A.trialnormals[:, 1] .* (A.Btrial * xfmm)))[:, 1]
        fmm_res2 =
            A.testnormals[:, 2] .*
            (A.fmm * (A.trialnormals[:, 2] .* (A.Btrial * xfmm)))[:, 1]
        fmm_res3 =
            A.testnormals[:, 3] .*
            (A.fmm * (A.trialnormals[:, 3] .* (A.Btrial * xfmm)))[:, 1]

        y .+= A.operator.alpha * A.Btest * (fmm_res1 + fmm_res2 + fmm_res3)
    end

    if A.operator.beta != 0.0
        fmm_curl1 = A.curlBtest[1] * (A.fmm * (A.curlBtrial[1] * xfmm))[:, 1]
        fmm_curl2 = A.curlBtest[2] * (A.fmm * (A.curlBtrial[2] * xfmm))[:, 1]
        fmm_curl3 = A.curlBtest[3] * (A.fmm * (A.curlBtrial[3] * xfmm))[:, 1]

        y .+= A.operator.beta .* (fmm_curl1 + fmm_curl2 + fmm_curl3)
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:CFMMMatrixHH3DHyperSingular},
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
    if A.operator.alpha != 0.0
        fmm_res1 =
            A.trialnormals[:, 1] .* (transpose(A.fmm) * (A.testnormals[:, 1] .* (xx)))[:, 1]
        fmm_res2 =
            A.trialnormals[:, 2] .* (transpose(A.fmm) * (A.testnormals[:, 2] .* (xx)))[:, 1]
        fmm_res3 =
            A.trialnormals[:, 3] .* (transpose(A.fmm) * (A.testnormals[:, 3] .* (xx)))[:, 1]

        y .+= A.operator.alpha * transpose(A.Btrial) * (fmm_res1 + fmm_res2 + fmm_res3)
    end

    if A.operator.beta != 0.0
        fmm_curl1 =
            transpose(A.curlBtrial[1]) *
            (transpose(A.fmm) * (transpose(A.curlBtest[1]) * xfmm))[:, 1]
        fmm_curl2 =
            transpose(A.curlBtrial[2]) *
            (transpose(A.fmm) * (transpose(A.curlBtest[2]) * xfmm))[:, 1]
        fmm_curl3 =
            transpose(A.curlBtrial[3]) *
            (transpose(A.fmm) * (transpose(A.curlBtest[3]) * xfmm))[:, 1]
        y .+= A.operator.beta .* (fmm_curl1 + fmm_curl2 + fmm_curl3)
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:CFMMMatrixHH3DHyperSingular},
    x::AbstractVector,
)
    mul!(y, transpose(adjoint(At)), conj(x))

    return conj!(y)
end
