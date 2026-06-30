# CorrectionFactorMatrixMethod.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoshuaTetzner.github.io/CorrectionFactorMatrixMethod.jl/stable/)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoshuaTetzner.github.io/CorrectionFactorMatrixMethod.jl/dev/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JoshuaTetzner/CorrectionFactorMatrixMethod.jl/blob/main/LICENSE)
[![CI](https://github.com/JoshuaTetzner/CorrectionFactorMatrixMethod.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JoshuaTetzner/CorrectionFactorMatrixMethod.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/JoshuaTetzner/CorrectionFactorMatrixMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoshuaTetzner/CorrectionFactorMatrixMethod.jl)

## Introduction

This package builds matrix-free boundary-element operators that combine a fast
multipole approximation of the far interactions with directly assembled
near-field corrections. The result behaves like the dense boundary-element
matrix under matrix-vector products, but is never assembled explicitly, so it
can be used directly inside an iterative solver such as `Krylov.gmres`.

The generic correction and linear-map machinery lives in the core package.
Support for concrete operators and spaces is provided through package
extensions for
[BEAST.jl](https://github.com/krcools/BEAST.jl),
[CompScienceMeshes.jl](https://github.com/krcools/CompScienceMeshes.jl), and
[ExaFMMt.jl](https://github.com/JoshuaTetzner/ExaFMMt.jl), which are loaded
automatically once these packages are available.

## Correction-Factor Matrix Method

For a boundary-element matrix $A$, the far field is approximated by an
FMM-backed map $A_\mathrm{FMM}$. The near interactions are evaluated with the
boundary-element quadrature and corrected for the part already represented by
the FMM:

$$Ax \approx A_\mathrm{FMM}\,x + \left(A_\mathrm{near} - A_\mathrm{FMM,near}\right)x.$$

The sparse correction blocks are selected from an
[H2Trees.jl](https://github.com/djukic14/H2Trees.jl) tree. A matrix-vector
product evaluates the FMM map first and then adds the sparse near correction.
Transpose and adjoint products follow the corresponding `LinearMaps.jl`
interfaces. Further details are given in the
[documentation](https://JoshuaTetzner.github.io/CorrectionFactorMatrixMethod.jl/dev/details/method/).

## Installation

Installing CorrectionFactorMatrixMethod is done by entering the package manager
(enter `]` at the Julia REPL) and issuing:

```
pkg> add https://github.com/JoshuaTetzner/CorrectionFactorMatrixMethod.jl.git
```

The supported boundary-element integration requires Julia 1.10 or later.

## First steps

Load the optional BEAST, CompScienceMeshes, and ExaFMMt packages, build a
boundary-element operator with its test and trial spaces, and assemble the
correction-factor operator:

```julia
using BEAST, CompScienceMeshes, ExaFMMt
using CorrectionFactorMatrixMethod

mesh = meshsphere(1.0, 0.4)
space = raviartthomas(mesh)
operator = Maxwell3D.singlelayer(; wavenumber=1.0)

matrix = CFMM.assemble(operator, space)
result = matrix * rand(scalartype(operator), numfunctions(space))
```

[`CFMM.assemble`](https://JoshuaTetzner.github.io/CorrectionFactorMatrixMethod.jl/dev/manual/manual/)
constructs an optimized tree automatically. Existing trees and all lower-level
FMM, quadrature, and scheduler options can be supplied as keywords. The
returned operator implements the `LinearMaps.jl` interface, so it can be passed
straight to an iterative solver. Runnable EFIE and MFIE examples are in the
[`examples/`](examples) directory and in the
[documentation](https://JoshuaTetzner.github.io/CorrectionFactorMatrixMethod.jl/dev/).

## References

- [1] Adelman, Ross, Nail A. Gumerov, and Ramani Duraiswami. *FMM/GPU-Accelerated Boundary Element Method for Computational Magnetics and Electrostatics.* IEEE Transactions on Magnetics 53, no. 12 (December 2017): 1–11. [https://doi.org/10.1109/TMAG.2017.2725951](https://doi.org/10.1109/TMAG.2017.2725951).
