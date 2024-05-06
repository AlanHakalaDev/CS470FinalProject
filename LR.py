import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Model learning function
def logisticRegression(trainData, trainLabel, trainingVal, validationLabel, learningRate=0.01, epochs=1000):
    # add a column of ones to trainData
    trainDataTrain = np.c_[np.ones((len(trainData), 1)), trainData]

    # initialize our weights m
    m = np.random.randn(trainDataTrain.shape[1], 1)

    # initialize vars for measuring performance
    bestModel = m
    bestPerformance = 0

    #iterate over epochs
    for epoch in range(epochs):
        #get predictied Y
        predY = sigmoid(np.dot(trainDataTrain, m))
        #calculate loss
        loss = -np.sum(trainLabel * np.log(predY) + (1 - trainLabel) * np.log(1 - predY)) / len(trainDataTrain)
        #further progress gradient descent
        gradientM = np.dot(trainDataTrain.T, (predY - trainLabel)) * 2 / len(trainDataTrain)

        #Model evaluation on validation dataset
        predYVal = sigmoid(np.dot(np.c_[np.ones((len(trainingVal), 1)), trainingVal], m))
        predLabelsVal = (predYVal > 0.5).astype(int)
        accuracyVal = accuracy_score(validationLabel, predLabelsVal)

        #check if we have a better model
        if accuracyVal > bestPerformance:
            #if so update it
            bestModel = m
            bestPerformance = accuracyVal
        
        #update m
        m -= learningRate * gradientM
    
    #return our best model and how well it did
    return bestModel, bestPerformance

# Load dataset
df = pd.read_csv("spambase.csv")

#split data into training and testing data
train, test = train_test_split(df, test_size=0.2, random_state = 69)

trainData = test.drop(['spam'], axis=1)
trainLabel = df['spam']

# Initialize K-fold cross-validation
fold5 = KFold(n_splits=5)

#set up vars to find best model
highestAccuracy = 0
bestK = 0

# Create KFold object
kf = KFold(n_splits=5)
bestAccuracy = 0
bestFoldYTest = None

# Iterate over the folds
print("Looping over 5 folds for cross fold validation")
for trainIndex, testIndex in kf.split(trainData):
    trainDataTrain, trainDataVal = trainData[trainIndex], trainData[testIndex]
    trainLabelTrain, trainLabelTest = trainLabel[trainIndex], trainLabel[testIndex]
    
    # Perform logistic regression
    bestModel, accuracyTest = logisticRegression(trainDataTrain, trainLabelTrain, trainDataVal, trainLabelTest)
    
    #update if we have a better model
    if accuracyTest > bestAccuracy:
        bestAccuracy = accuracyTest
        bestFoldYTest = trainLabelTest
    #print current best accuracy test for this fold
    print(f"Fold Accuracy: {accuracyTest} with y_test: {trainLabelTest}")

#print best overall
print(f"Highest Accuracy: {bestAccuracy} with y_test: {bestFoldYTest}")
