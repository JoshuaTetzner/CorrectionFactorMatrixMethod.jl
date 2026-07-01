@testset "CFMMBEAST extension" begin
    @test Base.get_extension(CFM, :CFMMBEAST) !== nothing

    mesh = meshrectangle(1.0, 1.0, 0.5)
    scalarspace = lagrangec0d1(mesh)
    vectorspace = raviartthomas(mesh)
    operator = Helmholtz3D.singlelayer(; wavenumber=0.5)
    nearquadstrat = CFM.defaultnearquadstrat(operator, scalarspace, scalarspace)
    farquadstrat = CFM.SafeDoubleNumQStrat(3, 3)

    @test CFM.scalartype(operator) == BEAST.scalartype(operator)
    @test CFM.alpha(operator) == operator.alpha
    @test CFM.beta(Helmholtz3D.hypersingular(; wavenumber=0.5)) ==
        Helmholtz3D.hypersingular(; wavenumber=0.5).beta

    matrix = CFM.AbstractCorrectedKernelMatrix(
        operator, scalarspace, scalarspace; nearquadstrat=nearquadstrat
    )
    block = zeros(ComplexF64, size(matrix))
    matrix(block, collect(axes(block, 1)), collect(axes(block, 2)))
    near = assemble(operator, scalarspace, scalarspace; quadstrat=nearquadstrat)
    far = assemble(operator, scalarspace, scalarspace; quadstrat=farquadstrat)
    @test block ≈ near - far

    for space in (scalarspace, vectorspace)
        points, quadrature = CFM.sources(space, 2)
        indices, values = CFM.potentials(quadrature, space)
        @test size(points, 2) == 3
        @test size(indices, 2) == 2
        @test size(values, 1) == size(indices, 1)
    end

    _, scalarquadrature = CFM.sources(scalarspace, 2)
    curlindices, curlvalues = CFM.curlpotentials(scalarquadrature, scalarspace)
    normalvalues = CFM.normals(scalarquadrature)
    @test size(curlvalues) == (size(curlindices, 1), 3)
    @test size(normalvalues, 2) == 3

    _, vectorquadrature = CFM.sources(vectorspace, 2)
    divindices, divvalues = CFM.divpotentials(vectorquadrature, vectorspace)
    @test length(divvalues) == size(divindices, 1)

    for (operator, space, matrixtype) in (
        (
            Helmholtz3D.singlelayer(; wavenumber=0.5),
            scalarspace,
            CFM.CFMMMatrixHH3DSingleLayer,
        ),
        (
            Helmholtz3D.doublelayer(; wavenumber=0.5),
            scalarspace,
            CFM.CFMMMatrixHH3DDoubleLayer,
        ),
        (
            Helmholtz3D.doublelayer_transposed(; wavenumber=0.5),
            scalarspace,
            CFM.CFMMMatrixHH3DDoubleLayerTransposed,
        ),
        (
            Helmholtz3D.hypersingular(; wavenumber=0.5),
            scalarspace,
            CFM.CFMMMatrixHH3DHyperSingular,
        ),
        (
            Maxwell3D.singlelayer(; wavenumber=0.5),
            vectorspace,
            CFM.CFMMMatrixMW3DSingleLayer,
        ),
        (
            Maxwell3D.doublelayer(; wavenumber=0.5),
            vectorspace,
            CFM.CFMMMatrixMW3DDoubleLayer,
        ),
    )
        _, quadrature = CFM.sources(space, 2)
        matrix = CFM.FMMFunctor()(
            operator,
            space,
            space,
            quadrature,
            quadrature,
            nothing;
            scheduler=CFM.SerialScheduler(),
        )
        @test matrix isa matrixtype
        @test size(matrix) == (length(space), length(space))
        @test size(matrix, 1) == length(space)
        @test size(matrix, 2) == length(space)
    end
end
