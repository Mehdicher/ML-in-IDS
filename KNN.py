import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 

data = pd.read_csv('Train_data.csv/Train_data.csv')

# Convert categorical variables into numerical ones
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {} #  Initializes an empty dictionary to store label encoders for each categorical column.
for col in categorical_columns:
    le = LabelEncoder()
# Fits the label encoder to the column and transforms the categorical values to numeric values, storing the result back in the DataFrame.    
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le # Stores the label encoder in the label_encoders dictionary.

# Split the training dataset into features and target variable
X = data.drop(columns='class')
y = data['class']

# Encode the target variable
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)
data.head()

# Define a function to split the data
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    
    # Sets the seed for NumPy’s random number generator to ensure reproducibility of the random split.
    # n is the total number of samples in the dataset.
    np.random.seed(random_state)
    n = len(X)
    
    # size=int(test_size * n): Number of indices to select, calculated as the product of test_size and n.
    # replace=False: Ensures that the same index is not selected more than once.
    test_indices = np.random.choice(n, size=int(test_size * n), replace=False)
    
    # np.setdiff1d(np.arange(n), test_indices): Returns the indices that are in the array np.arange(n) but not in test_indices.
    train_indices = np.setdiff1d(np.arange(n), test_indices)
    
    # Uses the indices to split the features (X) and target (y) into training and test sets.
    return X.iloc[train_indices], X.iloc[test_indices], y[train_indices], y[test_indices]

# Split the data using your custom function
X_train, X_test, y_train, y_test = train_test_split_custom(X, y_encoded, test_size=0.2, random_state=42)

class KNNClassifier:
    def __init__(self, k=3): #Initializes the classifier with the number of neighbors k
        self.k = k  # k: int
                    #The number of closest neighbors that will determine the class of the 
                    #sample that we wish to predict.
    
    def fit(self, X, y): #Fits the classifier to the training data
        #Stores the training features and labels.
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):#Predicts the class labels for the given data
        return [self._predict(x) for x in X] #Calls _predict for each sample and returns the predictions.
    
    def _predict(self, x):#Predicts the class label for a single sample
        
        # Ensure x is a numpy array and convert to float64
        x = np.array(x, dtype=np.float64)
        
        # Compute distances between x and all examples in the training set
        #Calculates the Euclidean distance between the sample and each training sample.
        # formule : distance = sqrt (sum(i=1 --> n)(x[i]-y[i])**2)
        #it calculates the Frobenius norm  using linalg.norm(x), which computes the square root of the sum of the squared values of all elements.
        # and computes the Euclidean/L2 norm of each column
        distances = [np.linalg.norm(x - np.array(x_train, dtype=np.float64)) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label among the k nearest neighbors.
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

  #The cross_validate function performs k-fold cross-validation for different values of k./
#It calculates and prints the mean accuracy for each value of k.

#Use KFold to split the dataset into k folds.
def cross_validate(X, y, model_class, k_values, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) # Initializes K-Fold cross-validation with n_splits folds.
    results = {}
    
    X = np.array(X)  # Convert DataFrame to NumPy array
    y = np.array(y)  # Convert Series to NumPy array
    
    for k in k_values: # Iterates over each value of k
        model = model_class(k=k)  # Initializes a model with the current value of k.
        accuracies = []    # Initializes an empty list to store accuracies.
        
        for train_index, test_index in kf.split(X): # Splits the data into training and testing sets.
            #Gets the training and testing features.
            X_train, X_test = X[train_index], X[test_index] # X[train_index] and X[test_index] are used for splitting data.
            #Gets the training and testing labels.
            y_train, y_test = y[train_index], y[test_index] # y[train_index] and y[test_index] are similarly used for target values
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)  # Predicts the class labels for the testing data.
            accuracy = accuracy_score(y_test, predictions) # Calculates the accuracy of the predictions.
            accuracies.append(accuracy) # Appends the accuracy to the list.
        
        mean_accuracy = np.mean(accuracies) # Calculates the mean accuracy for the current value of k
        results[k] = mean_accuracy # Stores the mean accuracy in the results dictionary.
        print(f'k={k}, Mean Accuracy: {mean_accuracy:.4f}') #Prints the mean accuracy for the current value of k.
    
    return results

# Hyperparameter tuning for KNN
#After cross-validation, the best value of k is selected based on the highest mean accuracy.

k_values = [3, 5, 7, 9, 11]  # Different values for k  to find the best-performing one.
results = cross_validate(X, y_encoded, KNNClassifier, k_values) #Performs cross-validation for each value of k.

# Find the best hyperparameter value
best_k = max(results, key=results.get) #Finds the best value of k based on the highest mean accuracy.
print(f'Best k: {best_k}, Accuracy: {results[best_k]:.4f}')

# Train the KNN model with the best hyperparameters
knn_best = KNNClassifier(k=best_k)
knn_best.fit(X_train, y_train)

# Make predictions on the test dataset
predictions_test = knn_best.predict(X_test)

# the KNN model is then trained with the best value of k and evaluated on the test set.
accuracy_test = accuracy_score(y_test, predictions_test)
print(f'Test Accuracy with Best k: {accuracy_test:.4f}')

# Initialize and train the KNN classifier
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)

# Make predictions on the training dataset
predictions_train = knn.predict(X_train.values)

# Evaluate accuracy on the training dataset
accuracy_train = np.mean(predictions_train == y_train)
print(f'Training Accuracy: {accuracy_train}')

# Make predictions on the test dataset
predictions_test = knn.predict(X_test.values)

# Evaluate accuracy on the test dataset
accuracy_test = np.mean(predictions_test == y_test)
print(f'Test Accuracy: {accuracy_test}')

# confusion_matrix

predictions_test = knn.predict(X_test.values)

cm = confusion_matrix(y_test, predictions_test)
cm

%matplotlib inline 
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

#classification_report 
print(classification_report(y_test, predictions_test))
