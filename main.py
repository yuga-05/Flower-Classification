import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

print("\nFirst 5 rows:\n", df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

sns.pairplot(df, hue='species')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=200)
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)

print("\nAccuracy Comparison:")
print("Logistic Regression:", lr_acc)
print("Decision Tree:", dt_acc)

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(y_test, lr_pred, "Logistic Regression CM")
plot_cm(y_test, dt_pred, "Decision Tree CM")

print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, lr_pred, target_names=iris.target_names))

print("\nMisclassification Insight:")
print("- Most errors occur between Versicolor and Virginica.")
print("- Setosa is almost always correctly classified.")
print("\n--- Predict Flower Species ---")
print("Enter values:")

try:
    sl = float(input("Sepal Length: "))
    sw = float(input("Sepal Width: "))
    pl = float(input("Petal Length: "))
    pw = float(input("Petal Width: "))

    sample = np.array([[sl, sw, pl, pw]])

    pred = lr.predict(sample)[0]
    print("\nPredicted Species:", iris.target_names[pred])

except:
    print("Invalid input!")