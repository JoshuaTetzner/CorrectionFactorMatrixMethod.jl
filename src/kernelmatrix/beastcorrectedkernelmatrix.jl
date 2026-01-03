struct BEASTCorrectedKernelMatrix{T,NearBlockAssemblerType,CorrectionBlockAssemblerType} <:
       AbstractCorrectedKernelMatrix{T}
    nearassembler::NearBlockAssemblerType
    correctionassembler::CorrectionBlockAssemblerType

    function BEASTCorrectedKernelMatrix{T}(nearassembler, correctionassembler) where {T}
        return new{T,typeof(nearassembler),typeof(correctionassembler)}(
            nearassembler, correctionassembler
        )
    end
end

function Base.size(M::BEASTCorrectedKernelMatrix, dim=nothing)
    if dim === nothing
        return (length(M.nearassembler.tfs), length(M.nearassembler.bfs))
    elseif dim == 1
        return length(M.nearassembler.tfs)
    elseif dim == 2
        return length(M.nearassembler.bfs)
    else
        error("dim must be either 1 or 2")
    end
end
