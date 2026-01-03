abstract type AbstractCorrectedKernelMatrix{T} end

function AbstractCorrectedKernelMatrix(operator, testspace, trialspace; args...) end

function (::AbstractCorrectedKernelMatrix)(tdata, sdata, matrixblock) end
Base.eltype(::AbstractCorrectedKernelMatrix{T}) where {T} = T
