import pandas as p
from sklearn.model_selection import KFold, train_test_split

def runNaiveBayes(trainData, trainLabel, laPlace, labelProbs):
    #find the probabilities of both classes
    if 0 not in labelProbs:
        labelProbs[0] = 0
    if 1 not in labelProbs:
        labelProbs[1] = 0

    #find how many unique words we could possibly have
    vocab = len(trainData.columns)
    
    #in our model we will calculate the probabilities of each frequency given each
    #possibility for label (spam or legit) after each email we attempt we store this list to determine
    #frequencies from which email yield the best result
    legitProb = [[0 for _ in range(vocab)] for _ in range(len(trainLabel))]
    spamProb  = [[0 for _ in range(vocab)] for _ in range(len(trainLabel))]
    #create array to store 

    #store whether these weights say more about a legit or spam email
    #print(probLegit)

    #find the probabilities of each parameter given each classification
    for ind, row in trainData.iterrows():
        maxProb = -1;
        #iterate over labels
        for feature, value in row.items():
            #find the probabiliry of this event occuring given that label has also occured
            valProbLegit = value * labelProbs[0]
            valProbSpam = value * labelProbs[1]

            #do laplacean smoothing before saving
            valProbLegit = (valProbLegit + laPlace) / (labelProbs[0] + vocab)
            valProbSpam = (valProbSpam + laPlace) / (labelProbs[1] + vocab)

        #boundary safety check
        if(ind < len(legitProb)):
            #Store probabilities in the arrays for later comparison
            legitProb[ind][trainData.columns.get_loc(feature)] = valProbLegit
            spamProb[ind][trainData.columns.get_loc(feature)] = valProbSpam


    #at this point we have 2 arrays of probabilities for each row classifying it as either
    #legit or spam, we will return this data so the calling function for evaluation
    return legitProb, spamProb

def modelEval(data, model):
    #unpack data from model tuple
    legitProb = model[2][0]
    spamProb = model[2][1]
    legitArr = model[2]
    spamArr = model[3]

    #iterate over rows
    for ind in range(0, len(legitArr)):
        #calculate running product
        legitProb *= data[ind] * legitArr[ind]
        spamProb *= data[ind] * spamArr[ind]

    
    if legitProb > spamProb:
        return (legitProb, 0)
    else:
        return (spamProb, 1) 

def naiveBayes(trainData, trainLabel, valData, valLabel):
    #these will determine the model specs
    highestAccuracy = 0
    idealLaPlace = 0
    #find probabilities for each label
    labelProbs = trainLabel.value_counts(normalize=True)


    #iterate over laplacean alpha parameters
    for laPlace in range(1, 11):
        #run the naive bayes algorithm
        legitProb, spamProb = runNaiveBayes(trainData, trainLabel, laPlace, labelProbs)
        #save model specs
        model = (legitProb, spamProb, probLabels, laPlace)
        #evaluate the accuracy of the model
        accuracyResult = modelEval(valData, model)

        if(highestAccuracy < accuracyResult[0]):
            highestAccuracy = accuracyResult[0]

def performance(data, model, label):
    #initialize counters
    correctCount = 0
    totalCount = len(data)
    
    for ind, row in data.iterrows():
        # Evaluate the model on this row
        result = modelEval(row, model)
        predictedLabel = result[1]
        actualLabel = label.iloc[ind]
        
        # Check if the prediction is correct
        if predictedLabel == actualLabel:
            correctCount += 1
    
    # Calculate the accuracy
    accuracy = correctCount / totalCount

    #return our result
    return accuracy


#create pandas dataframe for our csv file
df = p.read_csv("spambase.csv")

#for naive bayes do NOT include the last four (capitals data and label)
data = df.drop(['capital_run_length_average'], axis = 1)
data = data.drop(['capital_run_length_longest'], axis = 1)
data = data.drop(['capital_run_length_total'], axis = 1)

#split data into training and testing data
train, test = train_test_split(data, test_size=0.2, random_state = 69)

#get data and labels from train and testing data
trainData = train.drop(['spam'], axis = 1)
trainLabel = train['spam']

testData = test.drop(['spam'], axis = 1) 
testLabel = test['spam'] 

#display head
print(testData.head())


#initialize 5-fold validation
fold5 = KFold(5)

bestModel = 0
highestAccuracy = 0

#iterate over our 5 folds
print("Looping over 5 folds for cross fold validation")
for trainIdx, testIdx in fold5.split(trainData):
    #extract data from current indices
    dataTrain, dataVal = trainData.iloc[trainIdx], trainData.iloc[testIdx]
    labelTrain, labelVal = trainLabel.iloc[trainIdx], trainLabel.iloc[testIdx]
    #run this fold through naive bayes
    clf = naiveBayes(dataTrain, dataVal, labelTrain, labelVal)
    print("New model from fold " + str(trainIdx) + " created")
    #evaluate performance
    accuracy = performance(trainData, clf, trainLabel)

    #check if this model is an improvement
    if(highestAccuracy < accuracy):
        #assign this model to be the best
        bestModel = clf
        highestAccuracy = accuracy

print("The following model was determined to be the best: ")
print(clf)
print("with an accuracy of: ")
print(accuracy)


                
                
        
    
