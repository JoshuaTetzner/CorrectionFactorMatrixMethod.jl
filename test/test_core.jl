using H2Trees
using LinearMaps
using StaticArrays

const CFM = CorrectionFactorMatrixMethod

struct TestCorrectedKernelMatrix{T} <: CFM.AbstractCorrectedKernelMatrix{T}
    data::Matrix{T}
end

Base.size(matrix::TestCorrectedKernelMatrix, dim...) = size(matrix.data, dim...)

function (matrix::TestCorrectedKernelMatrix)(block, testindices, trialindices)
    block .= view(matrix.data, testindices, trialindices)
    return nothing
end

struct TestAssembler
    tfs::Vector{Int}
    bfs::Vector{Int}
end

struct TestCorrectionOperator
    farquadstrat
end

struct TestFMMFunctor <: CFM.FMMFunctor end

# Minimal concrete subtype with no callable override — exercises the base stub.
struct MinimalKernelMatrix{T} <: CFM.AbstractCorrectedKernelMatrix{T} end
Base.size(::MinimalKernelMatrix) = (0, 0)

CFM.scalartype(::TestCorrectionOperator) = Float64

function CFM.AbstractCorrectedKernelMatrix(
    operator::TestCorrectionOperator, testspace, trialspace; nearquadstrat, farquadstrat
)
    @assert farquadstrat === operator.farquadstrat
    data = [10.0i + j for i in eachindex(testspace), j in eachindex(trialspace)]
    return TestCorrectedKernelMatrix(data)
end

function (::TestFMMFunctor)(operator, testspace, trialspace; kwargs...)
    return LinearMap(zeros(CFM.scalartype(operator), length(testspace), length(trialspace)))
end

@testset "Core" begin
    quadrature = CFM.defaultfarquadstrat(nothing, nothing, nothing)
    @test quadrature isa CFM.SafeDoubleNumQStrat
    @test (quadrature.outer_rule, quadrature.inner_rule) == (3, 3)

    data = [1.0 2.0 3.0; 4.0 5.0 6.0]
    kernelmatrix = TestCorrectedKernelMatrix(data)
    block = zeros(2, 2)
    kernelmatrix(block, [2, 1], [1, 3])
    @test eltype(kernelmatrix) == Float64
    @test size(kernelmatrix) == size(data)
    @test block == [4.0 6.0; 1.0 3.0]

    assembler = TestAssembler(collect(1:3), collect(1:4))
    beastmatrix = CFM.BEASTCorrectedKernelMatrix{ComplexF64}(assembler, assembler)
    @test eltype(beastmatrix) == ComplexF64
    @test size(beastmatrix) == (3, 4)
    @test size(beastmatrix, 1) == 3
    @test size(beastmatrix, 2) == 4
    @test_throws ErrorException size(beastmatrix, 3)

    testspace = [SVector(x, 0.0, 0.0) for x in 0.0:3.0]
    trialspace = [SVector(x, 0.0, 0.0) for x in 0.5:1.0:2.5]
    tree = TwoNTree(testspace, trialspace, 0.25; testminvalues=1, trialminvalues=1)
    farquadstrat = CFM.SafeDoubleNumQStrat(2, 3)
    correctednears = CFM.assemblecorrectednears(
        TestCorrectionOperator(farquadstrat),
        testspace,
        trialspace,
        tree;
        nearquadstrat=nothing,
        farquadstrat=farquadstrat,
        scheduler=CFM.SerialScheduler(),
        isnear=(args...) -> true,
    )
    dense = [10.0i + j for i in eachindex(testspace), j in eachindex(trialspace)]
    @test correctednears * ones(length(trialspace)) == dense * ones(length(trialspace))

    assembled = CFMM.assemble(
        TestCorrectionOperator(farquadstrat),
        testspace,
        trialspace;
        fmmfunctor=TestFMMFunctor(),
        minhalfsize=0.25,
        testminvalues=1,
        trialminvalues=1,
        nearquadstrat=nothing,
        farquadstrat=farquadstrat,
        scheduler=CFM.SerialScheduler(),
        isnear=(args...) -> true,
    )
    @test assembled * ones(length(trialspace)) == dense * ones(length(trialspace))
    @test CFM.defaultminhalfsize(testspace, trialspace) == 1.5 / 2^10

    options = CFM.FMMFunctor(; p=10, ncrit=64)
    @test (options.p, options.ncrit) == (10, 64)
    @test CFM.defaultminvalues(options) == 64
    @test CFM.defaultminvalues(TestFMMFunctor()) == 50  # base FMMFunctor fallback
    matrix = [2.0 1.0; -1.0 3.0]
    vector = [1.0, 2.0]
    @test CFM.FMM(matrix) * vector == matrix * vector
    @test transpose(CFM.FMM(matrix, transpose(matrix))) * vector ==
        transpose(matrix) * vector
    # Adjoint is not implemented; use complex types to exercise the error branch
    # (for real element types LinearMaps silently routes adjoint through transpose)
    cmatrix = ComplexF64[2.0 1.0; -1.0 3.0]
    cvector = ComplexF64[1.0, 2.0]
    @test_throws ErrorException adjoint(CFM.FMM(cmatrix, transpose(cmatrix))) * cvector
    @test_throws ErrorException adjoint(CFM.FMM(cmatrix)) * cvector

    far = LinearMap(matrix)
    correction = LinearMap([0.5 -0.5; 1.0 0.0])
    cfmm = CFM.PetrovGalerkinCFMM{Float64}(
        correction, far, size(far), CFM.SerialScheduler()
    )
    @test cfmm * vector == (far + correction) * vector
    @test transpose(cfmm) * vector == transpose(far + correction) * vector
    @test adjoint(cfmm) * vector == adjoint(far + correction) * vector

    # Stubs that error when no extension is loaded — verify the error is thrown.
    @test isnothing(CFM.AbstractCorrectedKernelMatrix(nothing, nothing, nothing))
    @test isnothing(MinimalKernelMatrix{Float64}()(nothing, nothing, nothing))
    @test isnothing(CFM.scalartype(nothing))
    @test_throws ErrorException CFM.defaultnearquadstrat(nothing, nothing, nothing)
    @test_throws ErrorException CFM.FMMFunctor()(nothing)
    @test CFM.alpha(nothing) == 1.0
    @test CFM.beta(nothing) == 1.0
    @test_throws ErrorException CFM.sources(nothing, 2)
    @test_throws ErrorException CFM.potentials(Matrix{Float64}(undef, 0, 0), nothing)
    @test_throws ErrorException CFM.curlpotentials(Matrix{Float64}(undef, 0, 0), nothing)
    @test_throws ErrorException CFM.divpotentials(Matrix{Float64}(undef, 0, 0), nothing)
    @test_throws ErrorException CFM.normals(nothing)
    @test_throws ErrorException CFM.setup(nothing, nothing, nothing)
end
