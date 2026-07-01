@testset "Operators" begin
    scalarproblems, vectorproblems = testproblems()
    k = 2π / 10

    for operator in (
        Helmholtz3D.singlelayer(; wavenumber=k),
        Helmholtz3D.doublelayer(; wavenumber=k),
        Helmholtz3D.doublelayer_transposed(; wavenumber=k),
        Helmholtz3D.hypersingular(; wavenumber=k),
    )
        testoperator(operator, scalarproblems)
    end

    for operator in
        (Maxwell3D.singlelayer(; wavenumber=k), Maxwell3D.doublelayer(; wavenumber=k))
        testoperator(operator, vectorproblems)
    end

    problem = last(vectorproblems)
    operator = Maxwell3D.singlelayer(; wavenumber=k)
    matrix = CFMM.assemble(operator, problem.testspace, problem.trialspace)
    dense = assemble(operator, problem.testspace, problem.trialspace)
    x = ones(ComplexF64, size(matrix, 2))
    @test isapprox(matrix * x, dense * x; rtol=problem.tolerance)

    # Static (Laplace, gamma=Val(0)) operators have Float64 scalar type.
    # testoperator uses ComplexF64 vectors, triggering the real/complex
    # multiplication branch in each HH3D operator.
    for operator in (
        Helmholtz3D.singlelayer(),
        Helmholtz3D.doublelayer(),
        Helmholtz3D.doublelayer_transposed(),
        Helmholtz3D.hypersingular(),
    )
        testoperator(operator, scalarproblems)
    end

    # Single-space convenience form assemble(op, space)
    scalar_problem = last(scalarproblems)
    square_matrix = CFMM.assemble(
        Helmholtz3D.singlelayer(; wavenumber=k), scalar_problem.testspace
    )
    @test size(square_matrix) ==
        (length(scalar_problem.testspace), length(scalar_problem.testspace))
end
