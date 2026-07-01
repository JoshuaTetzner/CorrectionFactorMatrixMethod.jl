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

# Warn locally if pre-rendered documentation plots are older than their source.
# Git sets identical mtimes on checkout so this only fires after a local edit.
let
    repo = dirname(@__DIR__)
    scripts = map(
        f -> joinpath(repo, "examples", f), ["efie.jl", "mfie.jl", "plotresults.jl"]
    )
    assets = joinpath(repo, "docs", "src", "assets", "examples")
    newest = maximum(mtime, scripts)
    for name in ["efie_results.html", "mfie_results.html"]
        html = joinpath(assets, name)
        isfile(html) &&
            mtime(html) < newest &&
            @warn "$name may be stale — run: julia --startup-file=no docs/render_examples.jl"
    end
end

@run_package_tests verbose = true
