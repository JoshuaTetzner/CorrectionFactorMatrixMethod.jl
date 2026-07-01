using CorrectionFactorMatrixMethod
using Documenter

DocMeta.setdocmeta!(
    CorrectionFactorMatrixMethod,
    :DocTestSetup,
    :(using CorrectionFactorMatrixMethod);
    recursive=true,
)

makedocs(;
    modules=[CorrectionFactorMatrixMethod, CorrectionFactorMatrixMethod.CFMM],
    authors="Joshua Tetzner",
    repo=Remotes.GitHub("JoshuaTetzner", "CorrectionFactorMatrixMethod.jl"),
    sitename="CorrectionFactorMatrixMethod.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaTetzner.github.io/CorrectionFactorMatrixMethod.jl",
        edit_link="dev",
        assets=String[],
    ),
    pages=[
        "Introduction" => "index.md",
        "Manual" =>
            ["General Usage" => "manual/manual.md", "Examples" => "manual/examples.md"],
        "Details" => [
            "Correction-Factor Method" => "details/method.md",
            "Supported Operators" => "details/operators.md",
            "Package Extensions" => "details/extensions.md",
        ],
        "Contributing" => "contributing.md",
        "API Reference" => "apiref.md",
    ],
    checkdocs=:exports,
)

for (root, _, files) in walkdir(joinpath(@__DIR__, "build"))
    for file in files
        endswith(file, ".dat") && rm(joinpath(root, file))
    end
end

deploydocs(;
    repo="github.com/JoshuaTetzner/CorrectionFactorMatrixMethod.jl.git",
    devbranch="dev",
    push_preview=false,
)
