import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin # Base classes for scikit-learn compatibility.
from sklearn.metrics import classification_report 


data = pd.read_csv('Train_data.csv/Train_data.csv')

# Convert categorical variables into numerical ones
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {} # Initializes an empty dictionary to store label encoders for each categorical column.
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col]) ## Fits the label encoder to the column and transforms the categorical values to numeric values, storing the result back in the DataFrame. 
    label_encoders[col] = le # Stores the label encoder in the label_encoders dictionary.

# Split the dataset into features and target variable
X = data.drop(columns='class')
y = data['class']

# Encode the target variable
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# Display the mapping of original labels to encoded values
class_mapping = dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_)))
print("Class Mapping: ", class_mapping)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Display the processed data
print(X.head())
print(y_encoded[:5])

class DecisionTree:
    # Initializes the tree with a maximum depth.
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y) # Grows the decision tree using the _grow_tree method and sets it as self.tree.

        #Defines the predict method to make predictions.
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X]) #Traverses the tree for each sample in X to make predictions.

    # Defines a private method to grow the tree recursively.
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape # Gets the number of samples and features.
        n_labels = len(np.unique(y)) # Gets the number of unique labels.
        
        # Checks if the maximum depth is reached, only one label remains, or no samples are left.
        if depth >= self.max_depth or n_labels == 1 or n_samples == 0:
            leaf_value = self._most_common_label(y) # Gets the most common label in y.
            return Node(value=leaf_value) # Returns a leaf node with the most common label.

        feat_idxs = np.random.choice(n_features, n_features, replace=False) # Randomly selects features without replacement.

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs) # Finds the best feature and threshold for splitting.
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh) # Splits the data based on the best feature and threshold.
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1) # Recursively grows the left subtree.
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1) # Recursively grows the right subtree.
        return Node(best_feat, best_thresh, left, right) # Returns a node with the best feature, threshold, left subtree, and right subtree.

    def _best_criteria(self, X, y, feat_idxs): # Defines a private method to find the best feature and threshold for splitting.
    
    # Information gain is always non-negative. If we start with a value that is less than any possible gain (such as -1),
    # it guarantees that the first calculated gain will replace the initial value,ensuring that we are always improving upon our initial value
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:  # Loops through each selected feature
            X_column = X[:, feat_idx]  # Gets the column of the current feature
            thresholds = np.unique(X_column)  # Gets the unique values of the feature.
            for threshold in thresholds: # Loops through each unique value.
                gain = self._information_gain(y, X_column, threshold) # Calculates the information gain for the current threshold.

                if gain > best_gain:
                    best_gain = gain   # Updates the best gain.
                    split_idx = feat_idx   # Updates the best split index.
                    split_thresh = threshold  # Updates the best split threshold.

        return split_idx, split_thresh # Returns the best split index and threshold.

    def _information_gain(self, y, X_column, split_thresh): # Defines a private method to calculate information gain.
        parent_entropy = self._gini(y)  # Calculates the Gini impurity of the parent node.
        left_idxs, right_idxs = self._split(X_column, split_thresh) # Splits the data based on the threshold.

        if len(left_idxs) == 0 or len(right_idxs) == 0: #Checks if either split is empty
            return 0

        n = len(y) # Gets the number of samples.
        n_l, n_r = len(left_idxs), len(right_idxs)  # Gets the number of samples in the left and right splits
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs]) # Calculates the Gini impurity of the left and right splits.
        
        # Calculates the weighted average(Combine child impurities.) of the Gini impurity of the child nodes.
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy # Calculates the information gain.
        return ig
    
    # Defines a private method to split the data based on a threshold.
    def _split(self, X_column, split_thresh):
        #np.argwhere = Find the indices of array elements that are non-zero, grouped by element.
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() # Gets the indices of samples that are less than or equal to the threshold.
        right_idxs = np.argwhere(X_column > split_thresh).flatten() # Gets the indices of samples that are greater than the threshold.
        return left_idxs, right_idxs

    def _gini(self, y):
        # np.bincount= Count number of occurrences of each value in array of non-negative ints.
        hist = np.bincount(y) # Counts the occurrences of each class.
        ps = hist / len(y) # Calculate the probability of each label.
        return 1 - np.sum(ps ** 2)  # Gini impurity formula: 1−∑ pi**2


    def _most_common_label(self, y):
        counter = Counter(y) # Counts the occurrences(تكرارات) of each label. 
        most_common = counter.most_common(1)[0][0] # Gets the most common label
        return most_common
    
    # Defines a private method to traverse the tree and make a prediction for a single sample.
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value # Returns the value of the leaf node.

        if x[node.feature] <= node.threshold:      # Checks if the feature value is less than or equal to the threshold.
            return self._traverse_tree(x, node.left)    # Recursively traverses the left subtree.
        return self._traverse_tree(x, node.right)    # Recursively traverses the right subtree.

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None # Returns True if the node has a value (is a leaf node), otherwise False.

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10): #Initializes the RandomForest with a specified number of trees and maximum depth.
        self.n_trees = n_trees 
        self.max_depth = max_depth
        self.trees = []  # Initializes an empty list to store the trees.

    def fit(self, X, y): # Defines the fit method to train the random forest.
        self.trees = []  # Initializes an empty list to store the trained trees.
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth) # Initializes a DecisionTree with the specified maximum depth.
            X_samp, y_samp = self._bootstrap_sample(X, y) # Creates a bootstrap sample of the data.
            tree.fit(X_samp, y_samp)   # Fits the decision tree to the bootstrap sample.
            self.trees.append(tree)  # Adds the trained tree to the list.

    def predict(self, X):
        #For each tree in self.trees, tree.predict(X) predicts the class labels for the input data X.
        tree_preds = np.array([tree.predict(X) for tree in self.trees]) # The list comprehension collects these predictions into a list, which is then converted into a NumPy array.
        # Rearrange the shape of tree_preds to facilitate voting.
        # This rearrangement is necessary because we need to consider the predictions for each sample across all 
        # trees together, rather than the predictions for each tree across all samples.
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # tree_preds initially has the shape (n_trees, n_samples). np.swapaxes(tree_preds, 0, 1) changes its shape to (n_samples, n_trees).
        # Determine the final predicted class label for each sample by majority vote.
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred) # np.array(y_pred) converts the list of final predicted class labels into a NumPy array.

    def _bootstrap_sample(self, X, y): # Defines a private method to create a bootstrap sample.
        n_samples = X.shape[0]  # Gets the number of samples (returns the number of rows)
        # Generate a bootstrap sample of indices.
        # The range of numbers to choose from (i.e., 0 to n_samples - 1).
        # the number of indices to generate.
        # Allows the same index to be chosen more than once (sampling with replacement).
        idxs = np.random.choice(n_samples, n_samples, replace=True) # Selects sample indices with replacement.
        return X[idxs], y[idxs]  # Returns a tuple (X_bootstrap, y_bootstrap) where X_bootstrap and y_bootstrap are the bootstrap samples of the input data.

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

  # Create a wrapper class that allows RandomizedSearchCV to interact with the hyperparameters without modifying the original class

#    This class wraps the custom RandomForest class and makes it compatible with scikit-learn's interface by 
#    inheriting from BaseEstimator and ClassifierMixin.
class RandomForestWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.model = RandomForest(n_trees=self.n_trees, max_depth=self.max_depth)
        
# fit Method: Trains the custom RandomForest model.
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
# predict Method: Uses the trained custom RandomForest model to make predictions.
    def predict(self, X):
        return self.model.predict(X)

  # Defined a parameter distribution (param_dist) for RandomizedSearchCV and hyperparameter tuning.
param_dist = {
    'n_trees': [10, 20, 30, 40, 50],
    'max_depth': [5, 10, 15, 20, 25]
}

# Custom scorer for the RandomForest class
#Uses the accuracy score to evaluate the model's performance.
def custom_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return accuracy_score(y, predictions)

# Initialize RandomizedSearchCV with custom scoring
random_search = RandomizedSearchCV(estimator=RandomForestWrapper(), param_distributions=param_dist, scoring=make_scorer(custom_scorer), cv=5, n_iter=10, random_state=42)

# Fit the randomized search to the data
random_search.fit(X_train.values, y_train)

# Get the best parameters and the best estimator and used to make predictions and evaluate the accuracy on the test data.
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_

print(f'Best Parameters: {best_params}')
print(f'Best Estimator: {best_estimator}')

# Make predictions on the test dataset with the best estimator
best_predictions = best_estimator.predict(X_test.values)

# Evaluate accuracy on the test dataset with the best estimator
best_accuracy = accuracy_score(y_test, best_predictions)
print(f'Test Accuracy with Best Estimator: {best_accuracy}')


print(classification_report(y_test, predictions))
