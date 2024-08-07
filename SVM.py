import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

data = pd.read_csv('Train_data.csv/Train_data.csv')
data.columns
data.shape
data["class"].value_counts()


# Convert categorical variables into numerical ones
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}  # Initializes an empty dictionary to store label encoders for each categorical column.
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col]) # Fits the label encoder to the column and transforms the categorical values to numeric values, storing the result back in the DataFrame. 
    label_encoders[col] = le   # Stores the label encoder in the label_encoders dictionary.

# Split the training dataset into features and target variable
X = data.drop(columns='class')
y = data['class']

# Encode the target variable
y_encoded = LabelEncoder().fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

class SVM:
    
    # Initializes the classifier with learning rate, regularization parameter, and number of iterations.
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        # Stores the regularization parameter. which controls the trade-off between maximizing the margin and minimizing the classification error.
        self.lambda_param = lambda_param 
        self.n_iters = n_iters
        self.w = None #  Initializes the weight vector (w) and bias (b) as None. model parameters are learned from the data
        self.b = None

    def fit(self, X, y): # Fits the classifier to the training data.
        n_samples, n_features = X.shape # Gets the number of samples and features in the dataset X.
        y_ = np.where(y <= 0, -1, 1) # Converts the target labels to -1 and 1, SVMs typically use these labels to represent the two classes.
        
        self.w = np.zeros(n_features) # Initializes the weight vector w with zeros. Its size is equal to the number of features.
        self.b = 0

        # The loop variable _ takes on each value in this sequence one by one, 
        # but the actual value is not used inside the loop body. The underscore indicates that the value is being ignored.
        
        # the outer loop: runs for self.n_iters iterations, controlling the number of times the model's parameters (weights and bias) are updated during training.
        for _ in range(self.n_iters):
# the inner loop: iterates over each sample in the dataset X, using idx as the index and x_i as the feature vector of the current sample.
            for idx, x_i in enumerate(X): 
                
                # Calculates whether the current sample satisfies the margin condition.
                # This condition checks if the sample is correctly classified with a margin of at least 1.
                # y_idx: Transformed label of the i-th sample. either -1 or 1
                # x _i: Feature vector of the i-th sample.
                # w: Weight vector.
                # b: Bias term.
                # Formula: y_idx(x_i ⋅ w − b)≥1
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 
                
                # Update Weights for Correctly Classified Samples:
                if condition: # Checks if the sample satisfies the margin condition. 
                    # This update applies the regularization term, which penalizes large weights to prevent overfitting.
                    self.w -= self.lr * (2 * self.lambda_param * self.w) # formula: w←w−η(2λw)
                
                # Update Weights and Bias for Misclassified Samples:
                # Executes if the sample does not satisfy the margin condition.
                else: 
                    # executed when the condition y_[idx] * (np.dot(x_i, self.w) - self.b) < 1 is met. 
                    # This means the current sample "x_i" is either within the margin boundary or misclassified, 
                    # and thus, the weights (self.w) and bias (self.b) need to be updated to correct this.
                    
                    # 2 * self.lambda_param * self.w: This term comes from the regularization part of the SVM 
                    # objective function. It penalizes large weights to prevent overfitting. The factor 2 is derived from 
                    # the derivative of the regularization term (which is typically \lambda * ||w||^2), 
                    # resulting in 2 * \lambda * w when differentiated with respect to w.
                    
                    # np.dot(x_i, y_[idx]): This computes the gradient of the hinge loss with respect to the weights. 
                    # Here, x_i is the feature vector of the current sample, and y_[idx] is its corresponding label 
                    # (transformed to -1 or 1). The dot product represents the influence of this sample on the weight vector.
                    
                    # Multiplying by self.lr scales this gradient according to the learning rate, 
                    # and subtracting it from self.w updates the weights in the direction that minimizes the loss.
                    
                    # formula: w←w−η(2λw−xi​yi)
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])) # gives the total gradient for this step
                    
                    # Adjusts the bias to shift the decision boundary in the correct direction for better classification of the current sample.
                    # self.lr * y_[idx]: This term adjusts the bias term to account for the misclassification or margin 
                    # violation. It shifts the decision boundary to better classify the current sample.
                    # formula : b←b−η​yi

                    self.b -= self.lr * y_[idx] 

    def predict(self, X):
        # Computes the linear combination of the input features and the learned weights minus the bias.
        # formula : X⋅w−b
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output) #Returns the sign of the linear output, which determines the predicted class.

# Train the SVM model
svm = SVM()
svm.fit(X_train.values, y_train)

# Make predictions on the test dataset
predictions = svm.predict(X_test.values)

# Evaluate accuracy on the test dataset
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy without Cross-Validation: {accuracy}')

# Function for cross-validation
def cross_validate(X, y, model, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42) # Initializes K-Fold cross-validation with k folds.
    accuracies = []

    for train_index, test_index in kf.split(X): # Splits the data into training and testing sets.
        X_train, X_test = X[train_index], X[test_index] # Gets the training and testing features.
        y_train, y_test = y[train_index], y[test_index] # Gets the training and testing labels.

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) #Calculates the accuracy of the predictions.
        accuracies.append(accuracy) #Appends the accuracy to the list.
    
    return np.mean(accuracies), np.std(accuracies)

# the SVM model using cross-validation
svm = SVM()
mean_accuracy, std_accuracy = cross_validate(X.values, y_encoded, svm, k=5)

# Train the best model on the entire training set
svm.fit(X_train.values, y_train)

# Make predictions on the test dataset
predictions = svm.predict(X_test.values)

# Evaluate accuracy on the test dataset
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy without Cross-Validation: {accuracy}')
print(f'Mean Cross-Validation Accuracy: {mean_accuracy}')
print(f'Standard Deviation of Cross-Validation Accuracy: {std_accuracy}')

# Hyperparameter tuning
#Iterate over different combinations of learning_rate, lambda_param, and n_iters.
#Track the best parameters based on mean accuracy.
learning_rates = [0.0001, 0.001, 0.01]
lambda_params = [0.001, 0.01, 0.1] # Defines a list of regularization parameters to test.
n_iters = [500, 1000, 2000]

best_accuracy = 0
best_params = {} # Initializes an empty dictionary to store the best parameters.

#Three nested loops iterate over all combinations of learning rates, lambda parameters, and iteration counts.
for lr in learning_rates:
    for lp in lambda_params:
        for ni in n_iters:
            # For each combination of hyperparameters, an instance of the SVM class is created with the current values of lr, lp, and ni.
            svm = SVM(learning_rate=lr, lambda_param=lp, n_iters=ni)
            
            # cross_validate function performs 5-fold cross-validation on the dataset X and y_encoded using the SVM model.
            mean_accuracy, std_accuracy = cross_validate(X.values, y_encoded, svm, k=5)
            print(f'Params: lr={lr}, lambda_param={lp}, n_iters={ni}, Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}')
            
            # If the current combination of hyperparameters results in a higher mean accuracy than the previously 
            # recorded best accuracy, update best_accuracy and best_params.
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = {'learning_rate': lr, 'lambda_param': lp, 'n_iters': ni}

print(f'Best Params: {best_params}, Best Accuracy: {best_accuracy:.4f}')

# Train the SVM model with best hyperparameters
best_svm = SVM(**best_params) #It allows you to dynamically pass parameters to a function or class without explicitly specifying each parameter.
best_svm.fit(X_train.values, y_train)

# Make predictions on the test dataset
predictions = best_svm.predict(X_test.values)

# Evaluate accuracy on the test dataset
accuracy = accuracy_score(y_test, predictions)
print(f'Test Accuracy with Best Hyperparameters: {accuracy:.4f}')
