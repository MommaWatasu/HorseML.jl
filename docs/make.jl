using Documenter
using HorseML
using HorseML.Preprocessing
using HorseML.NeuralNetwork
using HorseML.Regression
using HorseML.LossFunction
using HorseML.Classification
using HorseML.Tree

makedocs(;
    sitename="HorseML",
    pages = [
        "Home" => "index.md",
        "Tutorial" => [
            "Welcome to HorseML" => "Tutorial/Welcome.md",
            "Getting Started" => "Tutorial/Getting_Started.md",
            "Preprocessing" => "Tutorial/Preprocessing.md",
            "Classifiers" => "Tutorial/Classifiers.md",
            "Clustering" => "Tutorial/Clustering.md",
            "Tree" => "Tutorial/Tree.md",
            "NeuralNetwork" => "Tutorial/NeuralNetwork.md"
        ],
        "Manual" => [
            "Preprocessing" => "Manual/Preprocessing.md",
            "LossFunction" => "Manual/LossFunction.md",
            "Regression" => "Manual/Regression.md",
            "Classification" => "Manual/Classification.md",
            "Clustering" => "Manual/Clustering.md",
            "NeuralNetwork" => "Manual/NeuralNetwork.md",
        ]
    ]
)

deploydocs(
    repo = "github.com/MommaWatasu/HorseML.jl.git",
)