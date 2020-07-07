package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/ensemble"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	// Load in a dataset, with headers. Header attributes will be stored.
	// Think of instances as a Data Frame structure in R or Pandas.
	// You can also create instances from scratch.
	rawData, err := base.ParseCSVToInstances("titanic3.csv", true)
	if err != nil {
		panic(err)
	}

	// Print a pleasant summary of your data.
	fmt.Println(rawData)

	//Initialises knn and random forest classifier
	knnClf := knn.NewKnnClassifier("euclidean", "linear", 2)
	rfClf := ensemble.NewRandomForest(100, 4)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
	knnClf.Fit(trainData)
	rfClf.Fit(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictionsKNN, err := knnClf.Predict(testData)
	if err != nil {
		panic(err)
	}

	predictionsRF, err := rfClf.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Prints precision/recall metrics
	confusionMatKNN, err := evaluation.GetConfusionMatrix(testData, predictionsKNN)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}

	confusionMatRF, err := evaluation.GetConfusionMatrix(testData, predictionsRF)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println("=== KNN Model Evaluation ===")
	fmt.Println(evaluation.GetSummary(confusionMatKNN))

	fmt.Println("=== Random Forests Model Evaluation ===")
	fmt.Println(evaluation.GetSummary(confusionMatRF))
}
