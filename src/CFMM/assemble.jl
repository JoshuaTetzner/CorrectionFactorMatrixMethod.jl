function defaultminhalfsize(testspace::AbstractVector, trialspace::AbstractVector)
    _, testhalfsize = H2Trees.boundingbox(testspace)
    _, trialhalfsize = H2Trees.boundingbox(trialspace)
    return max(testhalfsize, trialhalfsize) / 2^10
end

defaultminvalues(::FMMFunctor) = 50
defaultminvalues(functor::ExaFMMtFunctor) = functor.ncrit

"""
    CFMM

High-level assembly interface for correction-factor FMM operators.
"""
module CFMM

using H2Trees
using ..CorrectionFactorMatrixMethod:
    FMMFunctor, PetrovGalerkinCFMM, defaultminhalfsize, defaultminvalues

"""
    assemble(operator, testspace, trialspace; kwargs...)

Assemble a matrix-free correction-factor FMM operator.

By default, an `H2Trees.TwoNTree` is constructed with a scale-dependent minimum
box size and a target leaf occupancy equal to the FMM `ncrit` setting. Pass
`tree` to use an existing tree, or override `minhalfsize`, `testminvalues`, and
`trialminvalues`.

All remaining keywords are forwarded to [`PetrovGalerkinCFMM`](@ref).
"""
function assemble(
    operator,
    testspace,
    trialspace;
    fmmfunctor=FMMFunctor(),
    tree=nothing,
    minhalfsize=nothing,
    testminvalues=defaultminvalues(fmmfunctor),
    trialminvalues=defaultminvalues(fmmfunctor),
    treeoptions=(;),
    kwargs...,
)
    if isnothing(tree)
        isnothing(minhalfsize) && (minhalfsize = defaultminhalfsize(testspace, trialspace))
        tree = H2Trees.TwoNTree(
            testspace,
            trialspace,
            minhalfsize;
            testminvalues=testminvalues,
            trialminvalues=trialminvalues,
            treeoptions...,
        )
    end
    return PetrovGalerkinCFMM(
        operator, testspace, trialspace, tree; fmmfunctor=fmmfunctor, kwargs...
    )
end

end
