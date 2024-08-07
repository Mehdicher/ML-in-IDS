# **PERFORMING the k-fold cross-validation AND hyperparameter optimization**

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV

data = pd.read_csv('Train_data.csv/Train_data.csv')

categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {} #  Initializes an empty dictionary to store label encoders for each categorical column.
for col in categorical_columns:
    le = LabelEncoder()
# Fits the label encoder to the column and transforms the categorical values to numeric values, storing the result back in the DataFrame.
    data[col] = le.fit_transform(data[col]) 
    label_encoders[col] = le # Stores the label encoder in the label_encoders dictionary.

X = data.drop(columns='class')
y = data['class']
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)
class_mapping = dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_)))
print("Class Mapping: ", class_mapping)

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    # Sets the seed for NumPy’s random number generator to ensure reproducibility of the random split.
    np.random.seed(random_state)
    # n is the total number of samples in the dataset.
    n = len(X)
    # n: Total number of samples.
    # size=int(test_size * n): Number of indices to select, calculated as the product of test_size and n.
    # replace=False: Ensures that the same index is not selected more than once.
    test_indices = np.random.choice(n, size=int(test_size * n), replace=False)
    # np.setdiff1d(np.arange(n), test_indices): Returns the indices that are in the array np.arange(n) but not in test_indices.
    train_indices = np.setdiff1d(np.arange(n), test_indices)
    # Uses the indices to split the features (X) and target (y) into training and test sets.
    return X.iloc[train_indices], X.iloc[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_custom(X, y_encoded, test_size=0.2, random_state=42)

def gini_impurity(y):
    unique_classes, counts = np.unique(y, return_counts=True) # Finds unique classes and their counts in y.
    probabilities = counts / len(y) # Calculates the probabilities of each class.
    return 1 - np.sum(probabilities ** 2) # Calculates and returns the Gini impurity using the formula Gini=1−∑(i=1 -> n) pi**2
                                          # where pi is the probability of class i.

class Node:
    def __init__(self, gini=None, num_samples=None, num_samples_per_class=None, predicted_class=None):
        self.gini = gini
        self.num_samples = num_samples # Stores the number of samples at the node.
        self.num_samples_per_class = num_samples_per_class # Stores the number of samples per class at the node.
        self.predicted_class = predicted_class # Stores the predicted class for the node.
        self.feature_index = 0 # Initializes the feature index used for splitting.
        self.threshold = 0 # Initializes the threshold value used for splitting.
        self.left = None
        self.right = None

# if max_depth is None, the tree will grow until all leaves are pure or until other stopping criteria are met.
# If it is set to a specific integer, the tree will stop growing when it reaches that depth.
class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth # Stores the maximum depth of the tree.
    
    def fit(self, X, y): # Fits the decision tree to the training data
        # Creates a set of unique elements from y, effectively finding all unique classes.
        self.n_classes_ = len(set(y)) # set(y): Creates a set of unique elements from the list y. And Counts the number of unique elements in y.
        self.n_features_ = X.shape[1] # Determines and stores the number of features in the feature matrix X.
        self.tree_ = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)] # Calculates the number of samples per class.
        # The numpy.argmax() function returns indices of the max element of the array in a particular axis.
        predicted_class = np.argmax(num_samples_per_class)# Returns the indices of the maximum values along an axis
        
        node = Node( # Creates a new Node with the calculated values.
            gini=gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        ) 
        
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y) # Finds the best feature and threshold to split the data.
            if idx is not None: # Checks if a valid split is found
                indices_left = X[:, idx] <= thr # Creates a boolean array indicating which samples go to the left child node.
                X_left, y_left = X[indices_left], y[indices_left]# Splits the data into left child node.
                X_right, y_right = X[~indices_left], y[~indices_left]# Splits the data into right child node.
                node.feature_index = idx # Sets the feature index for the split
                node.threshold = thr # Sets the threshold for the split.
                node.left = self._grow_tree(X_left, y_left, depth + 1) # Recursively grows the left child node.
                node.right = self._grow_tree(X_right, y_right, depth + 1) # Recursively grows the right child node.
        return node
    
# the threshold (thr) is a value used to split the data at each node of the tree. 
# The goal of finding the best threshold is to determine the point that most effectively 
# separates the data into distinct classes, thus reducing the impurity (such as Gini impurity) and creating more homogeneous subsets.    
    
    def _best_split(self, X, y): # The best split is the one that results in the greatest reduction in impurity.
        m, n = X.shape       # Gets the number of samples (m) and features (n).
        if m <= 1:       # If there is only one sample or less, no split is possible
            return None, None        # Returns None for both feature index and threshold
        
        best_idx, best_thr = None, None   # They will hold the index of the best feature and the best threshold value
        best_gini = 1                # Initializes the best Gini impurity as 1 (maximum impurity).
        for idx in range(n):              # The outer loop iterates over each feature (column) in the dataset 
            
            # sorts the data points by the current feature (X[:, idx]) and keeps the corresponding class labels (y).
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_ # Initializes the number of samples per class for the left node.
            
            # Calculates the number of samples per class for the right node.
            num_right = [np.sum(classes == i) for i in range(self.n_classes_)]
            
            for i in range(1, m): # the loop iterates over each possible split point (each sample)
                c = classes[i - 1]        # Gets the class of the current sample.
                num_left[c] += 1          # Increments the count for the left node.
                num_right[c] -= 1         # Decrements the count for the right node.
                
                # Calculates the Gini impurity for the left node and the right node
                gini_left = gini_impurity(classes[:i])
                gini_right = gini_impurity(classes[i:])
                # The Gini impurity for the split is the weighted average of the Gini impurities of the left and right subsets
                # i: The number of samples in the left subset.
                # m: The total number of samples in the dataset.
                # m - i: The number of samples in the right subset.
                # Dividing by "m" normalizes the sum to give an average impurity for the split.
                gini = (i * gini_left + (m - i) * gini_right) / m  # The weights are the proportions of samples in each subset.
                
                # If the calculated Gini impurity (gini) is lower than best_gini, update best_gini, best_idx, and best_thr.
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
# The best threshold is taken as the midpoint between the current value and the previous value of the sorted feature values
# thresholds[i]: The current value in the sorted list of feature values at index i.
# thresholds[i - 1]: The previous value in the sorted list of feature values.
# Adding these two values and dividing by 2 gives the midpoint, which is used as the threshold to split the data into two subsets.
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        
        # After evaluating all possible splits, the method returns the index of the best feature and the best threshold value.
        return best_idx, best_thr 
    
    
    
    def predict(self, X):       # Predicts the class for each sample in X.
        return [self._predict(inputs) for inputs in X] #Calls _predict for each sample and returns the predictions.
    
    def _predict(self, inputs): # Predicts the class for a single sample.
        node = self.tree_          # Starts at the root node.
        while node.left:        # While the current node has a left child
            # If the sample's feature value is less than or equal to the threshold, move to the left child.
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class # Returns the predicted class of the leaf node.
    
    def get_params(self, deep=True):
        return {"max_depth": self.max_depth}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Initialize a custom decision tree classifier
tree = CustomDecisionTreeClassifier(max_depth=5)

# Perform cross-validation

# Performs 5-fold cross-validation, calculating accuracy for each fold.
cv_scores = cross_val_score(tree, X.values, y_encoded, cv=5, scoring='accuracy') 
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {np.mean(cv_scores)}')

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10]
}

# Initializes a GridSearchCV with the custom decision tree classifier, parameter grid, 5-fold cross-validation, and accuracy scoring.
grid_search = GridSearchCV(CustomDecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X.values, y_encoded)# Fits the grid search to the data.
best_params = grid_search.best_params_ # Gets the best parameters from the grid search.and the best score
best_score = grid_search.best_score_
print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score}')

# Train and test with best parameters
best_tree = CustomDecisionTreeClassifier(**best_params) # Initializes a CustomDecisionTreeClassifier with the best parameters.
best_tree.fit(X_train.values, y_train) # Fits the classifier to the training data.
predictions_train_best = best_tree.predict(X_train.values)
accuracy_train_best = np.mean(predictions_train_best == y_train)
print(f'Best Training Accuracy: {accuracy_train_best}')
predictions_test_best = best_tree.predict(X_test.values)
accuracy_test_best = np.mean(predictions_test_best == y_test)
print(f'Best Test Accuracy: {accuracy_test_best}')

