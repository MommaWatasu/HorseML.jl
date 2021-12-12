@testset "utils" begin
    data = HorseML.sample(1:100, 30)
    @test ones(size(data)...) == (data .<= 100)
    data, ncdata = HorseML.sample(1:100, 30, nc=true)
    @test ones(size(data)...) == (data .<= 100)
    @test ones(size(ncdata)) == (ncdata .<= 100)
end