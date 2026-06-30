module CFMMExaFMMt

using CorrectionFactorMatrixMethod
using BEAST
using ExaFMMt

include("fmmoptions.jl")

function CorrectionFactorMatrixMethod.setup(spoints, tpoints, options::ExaFMMt.FMMOptions)
    return ExaFMMt.setup(spoints, tpoints, options)
end

function CorrectionFactorMatrixMethod.fmmresult(fmm::ExaFMMt.ExaFMM, x)
    return ExaFMMt.evaluate(fmm, x, fmm.fmmoptions)
end

end
