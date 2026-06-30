using Test
using TestItems
using TestItemRunner
using CorrectionFactorMatrixMethod

include("test_core.jl")

using BEAST
using CompScienceMeshes

include("CFMMBEAST/test_CFMMBEAST.jl")

using ExaFMMt

include("CFMMExaFMMt/test_CFMMExaFMMt.jl")

include("test_utils.jl")
include("test_operators.jl")

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
