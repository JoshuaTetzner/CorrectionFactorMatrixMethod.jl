# Supported Operators

All operators currently implemented target **three-dimensional** problems.
Two-dimensional support is a planned future extension.

The CFMM wrapper is constructed via [`CFMM.assemble`](@ref) (or the lower-level
[`PetrovGalerkinCFMM`](@ref)) once the `CFMMBEAST` and `CFMMExaFMMt` extensions
are loaded.

## Helmholtz 3D (scalar)

These operators act on scalar basis functions.
The current implementation supports **Lagrange C0D1** (`lagrangec0d1`) basis
functions on triangulated surfaces.

The Helmholtz Green's function is

```math
G(\bm{x},\bm{y}) = \frac{e^{ik|\bm{x}-\bm{y}|}}{4\pi|\bm{x}-\bm{y}|},
\quad k \in \mathbb{C}.
```

Setting `wavenumber = 0` recovers the Laplace Green's function
``G_0(\bm{x},\bm{y}) = 1/(4\pi|\bm{x}-\bm{y}|)``; the ExaFMMt backend
automatically switches to `LaplaceFMMOptions` in that case.

| Operator | BEAST constructor | Mathematical form |
|:---------|:------------------|:------------------|
| Single-layer (V) | `Helmholtz3D.singlelayer(; wavenumber)` | ``(V u)(\bm{x}) = \int_\Gamma G(\bm{x},\bm{y})\,u(\bm{y})\,\mathrm{d}S(\bm{y})`` |
| Double-layer (K) | `Helmholtz3D.doublelayer(; wavenumber)` | ``(K u)(\bm{x}) = \int_\Gamma \partial_{\bm{n}(\bm{y})} G(\bm{x},\bm{y})\,u(\bm{y})\,\mathrm{d}S(\bm{y})`` |
| Transposed double-layer (K') | `Helmholtz3D.doublelayer_transposed(; wavenumber)` | ``(K' u)(\bm{x}) = \int_\Gamma \partial_{\bm{n}(\bm{x})} G(\bm{x},\bm{y})\,u(\bm{y})\,\mathrm{d}S(\bm{y})`` |
| Hypersingular (W) | `Helmholtz3D.hypersingular(; wavenumber)` | ``(W u)(\bm{x}) = -\partial_{\bm{n}(\bm{x})} \int_\Gamma \partial_{\bm{n}(\bm{y})} G(\bm{x},\bm{y})\,u(\bm{y})\,\mathrm{d}S(\bm{y})`` |

All four operators are tested with `lagrangec0d1` basis functions (Petrov–Galerkin,
square systems). Other scalar basis function types may work but are not
currently verified.

## Maxwell 3D (vector)

These operators act on vector basis functions.
The current implementation supports **Raviart–Thomas** (`raviartthomas`) basis
functions on triangulated surfaces.

| Operator | BEAST constructor | Mathematical form |
|:---------|:------------------|:------------------|
| Single-layer / EFIE (T) | `Maxwell3D.singlelayer(; wavenumber)` | ``(\mathcal{T}\bm{u})(\bm{x}) = \alpha\!\int_\Gamma G\,\bm{u}\,\mathrm{d}S + \beta\,\nabla\!\int_\Gamma G\,\nabla_\Gamma\!\cdot\bm{u}\,\mathrm{d}S`` |
| Double-layer / MFIE (K) | `Maxwell3D.doublelayer(; wavenumber)` | ``(\mathcal{K}\bm{u})(\bm{x}) = \int_\Gamma \nabla G(\bm{x},\bm{y})\times\bm{u}(\bm{y})\,\mathrm{d}S(\bm{y})`` |

Both operators are tested with `raviartthomas` basis functions (Petrov–Galerkin,
square systems).

## Tested basis-function / operator combinations

The table below summarises which combinations are covered by the automated
test suite. A ✓ means the forward product, the transpose, and the adjoint
are all verified against a dense BEAST assembly.

| Operator | `lagrangec0d1` | `raviartthomas` |
|:---------|:--------------:|:---------------:|
| `Helmholtz3D.singlelayer` | ✓ | — |
| `Helmholtz3D.doublelayer` | ✓ | — |
| `Helmholtz3D.doublelayer_transposed` | ✓ | — |
| `Helmholtz3D.hypersingular` | ✓ | — |
| `Maxwell3D.singlelayer` | — | ✓ |
| `Maxwell3D.doublelayer` | — | ✓ |

Combinations marked — are not supported by the current operator
implementation (the scalar Helmholtz operators require scalar basis functions;
the vector Maxwell operators require vector basis functions).
