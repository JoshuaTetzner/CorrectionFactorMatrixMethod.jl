using Test
using TestItems
using TestItemRunner
using CorrectionFactorMatrixMethod

@testset "Core interfaces" begin
    CFMM = CorrectionFactorMatrixMethod

    options = CFMM.FMMFunctor(; p=10, ncrit=64)
    @test options.p == 10
    @test options.ncrit == 64

    quadrature = CFMM.defaultfarquadstrat(nothing, nothing, nothing)
    @test quadrature.outer_rule == 3
    @test quadrature.inner_rule == 3

    matrix = [2.0 1.0; -1.0 3.0]
    vector = [1.0, 2.0]
    @test CFMM.FMM(matrix) * vector ≈ matrix * vector
    @test transpose(CFMM.FMM(matrix, transpose(matrix))) * vector ≈
        transpose(matrix) * vector
end

@testitem "Code quality (Aqua.jl)" begin
    using Aqua
    using CorrectionFactorMatrixMethod

    Aqua.test_all(CorrectionFactorMatrixMethod)
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    using CorrectionFactorMatrixMethod

    @test JuliaFormatter.format(pkgdir(CorrectionFactorMatrixMethod); overwrite=false)
end

@testitem "Static analysis (JET.jl)" begin
    using JET
    using CorrectionFactorMatrixMethod

    JET.test_package(
        CorrectionFactorMatrixMethod; target_modules=(CorrectionFactorMatrixMethod,)
    )
end

@testitem "Explicit imports (ExplicitImports.jl)" begin
    using ExplicitImports
    using CorrectionFactorMatrixMethod

    @test ExplicitImports.check_no_stale_explicit_imports(CorrectionFactorMatrixMethod) ===
        nothing
    @test ExplicitImports.check_all_explicit_imports_via_owners(
        CorrectionFactorMatrixMethod
    ) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(CorrectionFactorMatrixMethod) ===
        nothing
end

@run_package_tests verbose = true
