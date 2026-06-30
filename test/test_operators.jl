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
end
