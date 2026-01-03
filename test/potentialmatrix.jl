using BEAST
using CompScienceMeshes

function sources(space::BEAST.Space, quadorder; dim=universedimension(space.geo))
    elements, _, _ = assemblydata(space)
    qp = BEAST.quadpoints(x -> refspace(space)(x), elements, (quadorder,))
    points = zeros(Float64, length(qp) * length(qp[1, 1]), dim)

    for (idx, pts) in enumerate(pts for element in qp for pts in element)
        points[idx, 1:dim] = pts.point.cart[1:dim]
    end

    return points, qp
end

using OhMyThreads
elements, _, _ = assemblydata(space)
qp = BEAST.quadpoints(x -> refspace(space)(x), elements, (3,))
OhMyThreads.@pmap for (idx, el) in enumerate(ts for el in qp for i in el)
    println(idx)
end

ind = 1
for el in qp
    for pts in el
        println(ind)
        ind += 1
    end
end

function potentialmatrix(qp::Q, X::BEAST.LagrangeBasis) where {Q}
    rfspace = refspace(X)
    _, tad, _ = assemblydata(X)
    len = length(qp) * length(qp[1, 1]) * size(tad.data)[1] * size(tad.data)[2]

    rc = ones(Int, len, 2)
    vals = zeros(Float64, len)
    sind = 1

    for (ncell, cell) in enumerate(qp[1, :])
        ind = (ncell - 1) * length(cell)
        for (npoint, point) in enumerate(cell)
            val = rfspace(point.point)
            for localbasis in eachindex(val)
                for data in tad.data[:, localbasis, ncell]
                    if data[1] != 0 && ind + npoint != 0
                        rc[sind, 1] = ind + npoint
                        rc[sind, 2] = data[1]
                        vals[sind] = val[localbasis].value * point.weight * data[2]
                        sind += 1
                    end
                end
            end
        end
    end

    return rc, vals
end

##
Γ = meshsphere(1.0, 0.08)

op = Helmholtz3D.singlelayer()
space = lagrangec0d1(Γ)
typeof(space)

points, qp = sources(space, 5)
