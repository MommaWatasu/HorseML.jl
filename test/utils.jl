using DataFrames
import HorseML.sample
import HorseML.@dataframe_func

@testset "utils" begin
    data = sample(1:100, 30)
    @test ones(size(data)...) == (data .<= 100)
    data, ncdata = sample(1:100, 30, nc=true)
    @test ones(size(data)...) == (data .<= 100)
    @test ones(size(ncdata)) == (ncdata .<= 100)

    @test_nowarn @dataframe_func function test(a::Int, x::AbstractMatrix; b::Int = 10)
        return (a.*x).+b
    end
end