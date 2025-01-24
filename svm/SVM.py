import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv("../data/diabetes.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

accuracy = []

for i in ["linear", "poly", "rbf", "sigmoid" ]:
    svm_model = SVC(kernel=i, random_state=42)
    svm_model.fit(X_train, y_train)
    pred_i = svm_model.predict(X_test)
    accuracy.append(accuracy_score(y_test, pred_i))
    print(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(["linear", "poly", "rbf", "sigmoid" ], accuracy, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. Algorithm')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.savefig('plot.png', dpi=800)