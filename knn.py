import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Function to calculate Euclidean distance between two points
def euclideanDist(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to find k nearest neighbors
def find_neighbors(train_data, trainLbl, testP, k):
    #create distance arr
    distances = []
    #enumerate over our training data
    for i, trainP in enumerate(train_data):
        #get euclidean distance
        dist = euclideanDist(trainP, testP)
        distances.append((trainLbl[i], dist))
    #sort distances
    distances.sort(key=lambda x: x[1])  # Sort distances
    #get neighbor count and return
    neighbors = [label for label, _ in distances[:k]]
    return neighbors

# Function to perform majority voting among neighbors
def majority_vote(neighbors):
    #quick bincount function
    counts = np.bincount(neighbors)
    return np.argmax(counts)

# K-nearest neighbors classifier
def knn(train_data, trainLbl, test_data, k):
    #keep track of predictions we make
    predictions = []
    #iterate over test data
    for testP in test_data:
        #find how many neighbors we have
        neighbors = find_neighbors(train_data, trainLbl, testP, k)
        #predict what label we will recieve for this amount of neighbors
        predicted_label = majority_vote(neighbors)
        #save our prediction
        predictions.append(predicted_label)
    return predictions

# Function to perform hyperparameter tuning for KNN
def hyperparameter_tuning(train_data, trainLbl, test_data, test_labels):
    print("Tweaking hyper parameters (kvalues)")
    # k values to try
    #k_values = [1, 5, 150, 42, 69]
    k_values = [1]

    #will determine which model is the best
    bestAcc = 0
    best_k = None

    #iterate over possible k values we're looking into
    for k in k_values:
        predictions = knn(train_data, trainLbl, test_data, k)
        accuracy = accuracy_score(test_labels, predictions)

        if accuracy > bestAcc:
            bestAcc = accuracy
            best_k = k
    
    return best_k, accuracy

def modelEval(train_data, trainLbl, test_data, test_labels, k):
    predictions = knn(train_data, trainLbl, test_data, k)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

# Load dataset
df = pd.read_csv("spambase.csv")

# Exclude the last four columns (capitals data and label) for KNN
data = df.drop(['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam'], axis=1)
labels = df['spam']

# Initialize K-fold cross-validation
fold5 = KFold(n_splits=5)

#set up vars to find best model
highestAccuracy = 0
bestK = 0

#create counter for folds
foldCount = 0
# Iterate over K-fold cross-validation splits
print("Looping over 5 folds for cross fold validation")
for train_idx, test_idx in fold5.split(data):
    # Extract data from current indices
    data_train, data_val = data.iloc[train_idx].values, data.iloc[test_idx].values
    label_train, label_val = labels.iloc[train_idx].values, labels.iloc[test_idx].values
    
    # Perform hyperparameter tuning for KNN
    results = hyperparameter_tuning(data_train, label_train, data_val, label_val)
    currK = results[0]
    print(f"Best K for fold {foldCount}: {results[0]} with accuracy {results[1]}")
    foldCount += 1
    
    #test this model on the training data
    accuracy = modelEval(data_train, label_train, data_val, label_val, currK)
    
    #check for a new best model
    if( highestAccuracy < accuracy ):
        highestAccuracy = accuracy
        bestK = currK

print(f"Best K value: {bestK} with accuracy {highestAccuracy}")
