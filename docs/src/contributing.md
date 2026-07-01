# Contributing

## What is currently implemented

CorrectionFactorMatrixMethod.jl provides a matrix-free BEM framework for
**three-dimensional** problems. The following is in place:

- **Core machinery**: correction-factor assembly driven by an `H2Trees.BlockTree`,
  sparse near-field correction via `BlockSparseMatrices.jl`, and a
  `LinearMaps.jl`-compatible `PetrovGalerkinCFMM` operator supporting forward,
  transpose, and adjoint products.
- **BEAST extension** (`CFMMBEAST`): quadrature, collocation, and corrected-kernel
  assembly for all BEAST `IntegralOperator` types; specialisations for
  Helmholtz 3D (V, K, K', W) with Lagrange C0D1 basis functions and Maxwell 3D
  (T/EFIE, K/MFIE) with Raviart–Thomas basis functions.
- **ExaFMMt extension** (`CFMMExaFMMt`): operator-parameter mapping to
  `LaplaceFMMOptions`, `ModifiedHelmholtzFMMOptions`, and `HelmholtzFMMOptions`;
  Helmholtz operators with zero wavenumber automatically reduce to the Laplace
  path.

For a complete list of tested operator–basis combinations see the
[Supported Operators](@ref) page.

## What could be added

Contributions are welcome in any of the following areas:

### Additional FMM backends

The `FMMFunctor` interface is designed to be extended. Adding a new FMM
library requires implementing:

- `CorrectionFactorMatrixMethod.setup(spoints, tpoints, options)` — set up the
  FMM from source and target point matrices.
- `CorrectionFactorMatrixMethod.fmmresult(fmm, x)` — evaluate the FMM
  map for a given input vector.

Potential targets include:

- **A native Julia FMM**: A pure-Julia implementation would remove the binary
  dependency on ExaFMMt and simplify installation, particularly on HPC
  systems.
- **FMMLIB2D / FMMLIB3D**: The Fortran-based FMM libraries wrapped for Julia
  could serve as an alternative backend.
- **GPU-accelerated FMMs**: An extension that drives a GPU FMM (e.g. via CUDA)
  would be a natural fit for the GPU near-field assembler already used for
  BEAST.

### Two-dimensional operator support

The current collocation and operator implementations target 3D surfaces only.
Extending to 2D would require:

- A 2D `sources` / `potentials` implementation for 1D boundary curves.
- Operators for the 2D Helmholtz and Laplace kernels
  (``G(\bm{x},\bm{y}) = \tfrac{i}{4} H_0^{(1)}(k|\bm{x}-\bm{y}|)``).
- A 2D-capable FMM backend (e.g. FMMLIB2D).

### Additional BEM library integrations

The `CFMMBEAST` extension can serve as a blueprint for integrating other
Julia BEM libraries. Any library that can provide quadrature data and
block assemblers in the same style can be added as a new extension.

### Further operator–basis combinations

Some operator–basis pairings (e.g. Helmholtz operators with
Raviart–Thomas basis functions, or Maxwell operators with Lagrange
basis functions for mixed formulations) are architecturally supported but not
yet tested. Contributions that add and verify such combinations are welcome.

## Regenerating documentation plots

The interactive plots on the Examples page are pre-rendered HTML files stored
in `docs/src/assets/examples/` and committed to the repository. The CI build
only serves these static files; it does not re-run the examples.

After changing any file in `examples/`, regenerate the plots from the
repository root:

```sh
julia --startup-file=no docs/render_examples.jl
```

Then commit the updated HTML files alongside your example changes. Running the
test suite locally will emit a `@warn` if the HTML files are older than the
example scripts, so you will not accidentally forget this step.

## Development workflow

Development targets the `dev` branch. Open a pull request from `dev` to
`main` for stable releases and merge with a merge commit.

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
the test suite. The formatter is configured with the `blue` style at a column
margin of 92.
