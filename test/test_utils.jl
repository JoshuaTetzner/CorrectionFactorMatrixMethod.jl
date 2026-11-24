using H2Trees
using LinearAlgebra
using StaticArrays

function testoperator(operator, problems)
    for (; testspace, trialspace, tree, tolerance) in problems
        dense = assemble(operator, testspace, trialspace)
        cfmm = CFMM.assemble(operator, testspace, trialspace; tree)

        for matrix in (identity, transpose, adjoint), T in (ComplexF64, ComplexF32)
            n = size(matrix(cfmm), 2)
            x = T.(complex.((1:n) ./ n, (n:-1:1) ./ n))
            y = matrix(cfmm) * x

            @test eltype(y) == promote_type(eltype(cfmm), T)
            @test isapprox(y, matrix(dense) * x; rtol=tolerance)
        end
    end
end

# The meshes are tiny, so `minvalues=8` is needed for the trees to split below the root
# (H2Trees ≥ 0.4 no longer forces a first subdivision). For the far-separated problem the
# error is set by the order of the far-field quadrature `SafeDoubleNumQStrat(3, 3)` against
# BEAST's default reference quadrature, not by the FMM, so its tolerance carries some headroom.
function testproblems()
    source = meshrectangle(1.0, 1.0, 0.3)
    scalarspace = lagrangec0d1(source)
    vectorspace = raviartthomas(source)
    scalarproblems = NamedTuple[]
    vectorproblems = NamedTuple[]

    for (offset, tolerance) in
        ((SVector(3.0, 0.0, 1.0), 5.0e-4), (SVector(0.2, 0.0, 0.2), 2.0e-3))
        target = translate(source, offset)
        testscalar = lagrangec0d1(target)
        testvector = raviartthomas(target)
        push!(
            scalarproblems,
            (
                testspace=testscalar,
                trialspace=scalarspace,
                tree=CorrectionFactorMatrixMethod.blocktree(
                    testscalar, scalarspace, 0.1; testminvalues=8, trialminvalues=8
                ),
                tolerance=tolerance,
            ),
        )
        push!(
            vectorproblems,
            (
                testspace=testvector,
                trialspace=vectorspace,
                tree=CorrectionFactorMatrixMethod.blocktree(
                    testvector, vectorspace, 0.1; testminvalues=8, trialminvalues=8
                ),
                tolerance=tolerance,
            ),
        )
    end

    push!(
        scalarproblems,
        (
            testspace=scalarspace,
            trialspace=scalarspace,
            tree=CorrectionFactorMatrixMethod.blocktree(
                scalarspace, scalarspace, 0.1; testminvalues=8, trialminvalues=8
            ),
            tolerance=2.0e-4,
        ),
    )
    push!(
        vectorproblems,
        (
            testspace=vectorspace,
            trialspace=vectorspace,
            tree=CorrectionFactorMatrixMethod.blocktree(
                vectorspace, vectorspace, 0.1; testminvalues=8, trialminvalues=8
            ),
            tolerance=2.0e-4,
        ),
    )

    return scalarproblems, vectorproblems
end
