# Correction-Factor Method

For a boundary-element matrix ``A``, the package approximates the far field by
an FMM-backed map ``A_{\mathrm{FMM}}``. Near interactions are evaluated with
the boundary-element quadrature and corrected for the interactions already
represented by the FMM:

```math
A x \approx A_{\mathrm{FMM}}x +
\left(A_{\mathrm{near}} - A_{\mathrm{FMM,near}}\right)x.
```

The sparse correction blocks are selected from an `H2Trees.BlockTree`.
Matrix-vector products evaluate the FMM map first and then add the sparse near
correction. Transpose and adjoint products follow the corresponding
`LinearMaps.jl` interfaces.
