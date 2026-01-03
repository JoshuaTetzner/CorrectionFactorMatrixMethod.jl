using BEAST
using CompScienceMeshes
using H2Trees
using ExaFMMt
using CorrectionFactorMatrixMethod

Γ = meshsphere(1.0, 0.1)

op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)
tree = H2Trees.TwoNTree(space, space, 0.1)
##

cfmm = CorrectionFactorMatrixMethod.PetrovGalerkinCFMM(op, space, space, tree)
##
x = rand(size(cfmm, 2))
y = cfmm * x

#A = assemble(op, space, space)
using LinearAlgebra
y2 = A * x
norm(y - y2) / norm(y2)
