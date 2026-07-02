# Correction-Factor Method

For a boundary-element matrix ``A``, the package approximates the far field by
an FMM-backed map ``A_{\mathrm{FMM}}``. Near interactions are evaluated with
the boundary-element quadrature and corrected for the interactions already
represented by the FMM:

```math
\boldsymbol{A}\boldsymbol{x} \approx \boldsymbol{A}_{\mathrm{FMM}}\boldsymbol{x} +
\left(\boldsymbol{A}_{\mathrm{near}} - \boldsymbol{A}_{\mathrm{FMM,near}}\right)\boldsymbol{x}.
```

The sparse correction blocks are selected from an `H2Trees.BlockTree`.
Matrix-vector products evaluate the FMM map first and then add the sparse near
correction. Transpose and adjoint products follow the corresponding
`LinearMaps.jl` interfaces.
