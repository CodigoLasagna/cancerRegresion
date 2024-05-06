import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Cargar el dataset desde el archivo CSV
df = pd.read_csv('dataset.csv')

# Regresión Lineal Simple
print("\nRegresión Lineal Simple:")
X_simple = df['Deaths - Neoplasms - Sex: Both - Age: Age-standardized (Rate)'].values.reshape(-1, 1)
y_simple = df['Deaths - Neoplasms - Sex: Both - Age: All Ages (Rate)'].values.reshape(-1, 1)
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.3, random_state=42)
regression_simple = LinearRegression()
regression_simple.fit(X_train_simple, y_train_simple)
y_pred_simple = regression_simple.predict(X_test_simple)
plt.scatter(X_test_simple, y_test_simple, color='blue')
plt.plot(X_test_simple, y_pred_simple, color='red')
plt.xlabel('Age-standardized (Rate)')
plt.ylabel('All Ages (Rate)')
plt.title('Regresión Lineal Simple')
plt.show()

# Regresión Múltiple
print("\nRegresión Múltiple:")
X_multiple = df.drop(columns=["Entity", "Code", "Year", "Deaths - Neoplasms - Sex: Both - Age: All Ages (Number)"])
y_multiple = df["Deaths - Neoplasms - Sex: Both - Age: All Ages (Number)"]
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y_multiple, test_size=0.3, random_state=42)
regression_multiple = LinearRegression()
regression_multiple.fit(X_train_multiple, y_train_multiple)
y_pred_multiple = regression_multiple.predict(X_test_multiple)
print('Coefficients:', regression_multiple.coef_)
print('Intercept:', regression_multiple.intercept_)

# Clasificador KNN
print("\nClasificador KNN:")
X_knn = df.drop(columns=["Entity", "Code", "Year"])
y_knn = df["Year"]
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)
y_pred_knn = knn.predict(X_test_knn)
print("Accuracy:", metrics.accuracy_score(y_test_knn, y_pred_knn))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test_knn, y_pred_knn))
print('\nClassification Report\n')
print(classification_report(y_test_knn, y_pred_knn))
