using HorseML.Preprocessing
using HorseML.Classification
using HorseML.Tree
using CSV
import DataFrames

@testset "Classification" begin
    data = Matrix(dataloader("iris"))
    x, t = data[:, 1:4], data[:, 5]
    
    #The test for Decision Tree
    model = DecisionTree()
    @test_nowarn Tree.fit!(model, x, t)
    @test_nowarn Tree.predict(model, x)
    @test_nowarn MV("test.dot", model)
    
    #The test for Random forest
    model = RandomForest(10)
    @test_nowarn Tree.fit!(model, x, t)
    @test_nowarn Tree.predict(model, x)
    paths = ["test$(string(i)).dot" for i in 1 : 10]
    @test_nowarn MV(paths, model)
    
    LE = LabelEncoder()
    OHE = OneHotEncoder()
    t = LE(t)
    t = OHE(t)
    
    #The test for Logistic Regression
    model = Logistic(alpha = 0.1)
    @test_nowarn Classification.fit!(model, x, t)
    @test_nowarn Classification.predict(model, x)
    
    #The test for Support Vector machine Classification(One-Vs-Rest)
    model = SVC()
    @test_nowarn Classification.fit!(model, x, t)
    @test_nowarn Classification.predict(model, x)
end