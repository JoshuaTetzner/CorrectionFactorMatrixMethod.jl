abstract type AbstractCorrectedKernelMatrix{T} end

function AbstractCorrectedKernelMatrix(operator, testspace, trialspace; args...) end

function (::AbstractCorrectedKernelMatrix)(matrixblock, tdata, sdata) end
Base.eltype(::AbstractCorrectedKernelMatrix{T}) where {T} = T
