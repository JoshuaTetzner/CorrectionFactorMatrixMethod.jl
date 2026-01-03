using H2Trees
using CompScienceMeshes

m = meshsphere(1.0, 0.1)
tree = TwoNTree(vertices(m), 0.1; minvalues=60)