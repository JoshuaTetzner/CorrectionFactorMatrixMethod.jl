# Contributing

Development work targets the `dev` branch. Open a pull request from `dev` to
`main` for stable changes, and merge it with a merge commit.

Run the package checks locally with:

```julia
using Pkg
Pkg.test()
```

Build the documentation with:

```julia
using Pkg
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
include("docs/make.jl")
```

Code must pass the formatter, Aqua, JET, and ExplicitImports checks included in
the test suite.
