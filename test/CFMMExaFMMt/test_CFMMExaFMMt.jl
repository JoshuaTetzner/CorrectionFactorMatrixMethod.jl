@testset "CFMMExaFMMt extension" begin
    @test Base.get_extension(CFM, :CFMMExaFMMt) !== nothing

    functor = CFM.FMMFunctor(; p=8, ncrit=32)
    operator = Helmholtz3D.singlelayer(; wavenumber=0.5)
    options = functor(operator)
    @test options isa ExaFMMt.FMMOptions
    @test (options.p, options.ncrit) == (8, 32)

    # Helmholtz with wavenumber=0 must reduce to the Laplace path
    @test functor(Helmholtz3D.singlelayer(; wavenumber=0.0)) isa ExaFMMt.LaplaceFMMOptions
    @test functor(Helmholtz3D.hypersingular(; wavenumber=0.0)) isa ExaFMMt.LaplaceFMMOptions

    sources = [0.0 0.0 0.0; 1.0 0.0 0.0]
    targets = [0.0 1.0 0.0; 1.0 1.0 0.0]
    fmm = CFM.setup(sources, targets, options)
    strengths = ones(ComplexF64, size(sources, 1))
    result = CFM.fmmresult(fmm, strengths)
    wrapped = CFM.setup_fmm(sources, targets, options; computetransposeadjoint=true)

    @test fmm isa ExaFMMt.ExaFMM
    @test result == ExaFMMt.evaluate(fmm, strengths, fmm.fmmoptions)
    @test size(result) == (size(targets, 1), 4)
    @test wrapped isa CFM.FMM
    @test CFM.fmmresult(wrapped, strengths) == result
end
