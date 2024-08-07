import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

df = pd.read_csv('Train_data.csv/Train_data.csv')
df.head()

# Load the dataset
data = pd.read_csv('Train_data.csv/Train_data.csv')

# Convert categorical variables into numerical ones
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split the dataset into features and target variable
X = data.drop(columns='class')
y = data['class']

# Encode the target variable
y = LabelEncoder().fit_transform(y)

X.head(), y[:5] #(1 is normal and 0 is anomaly) 

#This line displays the first few rows of the features (X) and the first five elements of the encoded target variable (y). 
#The output  (array([1, 1, 0, 1, 1], dtype=int64)) represents the first five encoded values of the target variable.

# Define a function to split the data
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    n = len(X)
    test_indices = np.random.choice(n, size=int(test_size * n), replace=False)
    train_indices = np.setdiff1d(np.arange(n), test_indices)
    return X.iloc[train_indices], X.iloc[test_indices], y[train_indices], y[test_indices]

# Split the data using your custom function
X_train, X_test, y_train, y_test = train_test_split_custom(X, y_encoded, test_size=0.2, random_state=42)

def gini_impurity(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

class Node:
    def __init__(self, gini=None, num_samples=None, num_samples_per_class=None, predicted_class=None):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None
        
        best_idx, best_thr = None, None
        best_gini = 1
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = [np.sum(classes == i) for i in range(self.n_classes_)]
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = gini_impurity(classes[:i])
                gini_right = gini_impurity(classes[i:])
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]
    
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

# Initialize and train the decision tree
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train.values, y_train)

# Make predictions on the training dataset
predictions_train = tree.predict(X_train.values)

# Evaluate accuracy on the training dataset
accuracy_train = np.mean(predictions_train == y_train)
print(f'Training Accuracy: {accuracy_train}')

# Make predictions on the test dataset
predictions_test = tree.predict(X_test.values)

# Evaluate accuracy on the test dataset
accuracy_test = np.mean(predictions_test == y_test)
print(f'Test Accuracy: {accuracy_test}')



