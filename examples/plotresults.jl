function plotresults(u, space, excitation, wavenumber, filename)
    Φ = [0.0]
    Θ = range(0; stop=π, length=100)
    points = [point(cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]
    farfield = potential(MWFarField3D(; wavenumber), points, u, space)
    currents, geometry = facecurrents(u, space)

    ys = range(-2; stop=2, length=50)
    zs = range(-4; stop=4, length=100)
    gridpoints = [point(0, y, z) for y in ys, z in zs]
    scatteredfield = potential(MWSingleLayerField3D(; wavenumber), gridpoints, u, space)
    incidentfield = excitation.(gridpoints)

    plot = Plot(
        Layout(
            Subplots(;
                rows=2,
                cols=2,
                specs=[Spec() Spec(; rowspan=2); Spec(; kind="mesh3d") missing],
            ),
        ),
    )
    add_trace!(plot, scatter(; x=Θ, y=norm.(farfield)); row=1, col=1)
    add_trace!(
        plot,
        contour(;
            x=zs,
            y=ys,
            z=norm.(scatteredfield - incidentfield)',
            colorscale="Viridis",
            zmin=0,
            zmax=2,
            showscale=false,
        );
        row=1,
        col=2,
    )
    add_trace!(plot, patch(geometry, norm.(currents); caxis=(0, 2)); row=2, col=1)
    savefig(plot, filename)
    return plot
end
