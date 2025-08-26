# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset as a DataFrame
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print(X.head())
print(y.head())

# Add species names to DataFrame
species = np.array(iris.target_names)
df = X.copy()
df["species"] = species[y]
print(df.head(10))

# Split data into train and test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Predictions:", y_pred_dt[:5])
print("True labels:", y_test[:5])
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree accuracy:", dt_accuracy)

# k-NN Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("k-NN accuracy:", knn_accuracy)