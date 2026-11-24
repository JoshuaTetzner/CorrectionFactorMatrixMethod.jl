function defaultminhalfsize(testspace::AbstractVector, trialspace::AbstractVector)
    _, testhalfsize = H2Trees.boundingbox(testspace)
    _, trialhalfsize = H2Trees.boundingbox(trialspace)
    return max(testhalfsize, trialhalfsize) / 2^10
end

defaultminvalues(::FMMFunctor) = 50
defaultminvalues(functor::ExaFMMtFunctor) = functor.ncrit

"""
    blocktree(testspace, trialspace, minhalfsize; testminvalues, trialminvalues, treeoptions...)

Build the `H2Trees.BlockTree` of two `H2Trees.TwoNTree`s used by the correction-factor
assembly.

By default no protrusion check is applied, matching the tree construction of H2Trees 0.3.
`treeoptions` are forwarded to both `H2Trees.TwoNTreeBuilder`s and override this default,
e.g. `protrusion=H2Trees.AutoProtrusionCheck()`.
"""
function blocktree(
    testspace,
    trialspace,
    minhalfsize;
    testminvalues::Int,
    trialminvalues::Int=testminvalues,
    treeoptions...,
)
    options = merge((; protrusion=H2Trees.NoProtrusionCheck()), values(treeoptions))
    testbuilder = H2Trees.TwoNTreeBuilder(;
        minhalfsize=minhalfsize, minvalues=testminvalues, options...
    )
    trialbuilder = H2Trees.TwoNTreeBuilder(;
        minhalfsize=minhalfsize, minvalues=trialminvalues, options...
    )
    return H2Trees.buildtree(
        testspace,
        trialspace;
        builder=H2Trees.BlockTreeBuilder(; test=testbuilder, trial=trialbuilder),
    )
end

"""
    CFMM

High-level assembly interface for correction-factor FMM operators.
"""
module CFMM

using ..CorrectionFactorMatrixMethod:
    FMMFunctor, PetrovGalerkinCFMM, blocktree, defaultminhalfsize, defaultminvalues

"""
    assemble(operator, testspace, trialspace; kwargs...)

Assemble a matrix-free correction-factor FMM operator.

By default, an `H2Trees.BlockTree` of two `H2Trees.TwoNTree`s is constructed with a
scale-dependent minimum box size and a target leaf occupancy equal to the FMM `ncrit`
setting. Pass `tree` to use an existing tree, or override `minhalfsize`, `testminvalues`,
and `trialminvalues`. `treeoptions` are forwarded to the `H2Trees.TwoNTreeBuilder`s.

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
        tree = blocktree(
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
