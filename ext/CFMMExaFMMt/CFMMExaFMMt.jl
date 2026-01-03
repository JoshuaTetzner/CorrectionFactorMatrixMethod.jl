module CFMMExaFMMt

using CorrectionFactorMatrixMethod
import CorrectionFactorMatrixMethod: ExaFMMtFunctor
using BEAST
using ExaFMMt

include("fmmoptions.jl")

function CorrectionFactorMatrixMethod.setup(spoints, tpoints, options::ExaFMMt.FMMOptions)
    return ExaFMMt.setup(spoints, tpoints, options)
end

end
