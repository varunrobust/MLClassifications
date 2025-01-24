import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv("../data/diabetes.csv")
# data = pd.read_csv("../data/pima-indians-diabetes.data.csv", names=columns)

X = data.iloc[:, :-1] 
y = data.iloc[:, -1]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

plt.figure(figsize=(16, 10))
plot_tree(dt_model, feature_names=columns[:-1], class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Visualization")
plt.savefig('plot.png', dpi=800)

max_depths = list(range(1, 21))
error_rates = []

for max_depth in max_depths:
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    error_rate = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error_rate)

plt.figure(figsize=(10, 6))
plt.plot(max_depths, error_rates, marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Effect of Max Depth on Error Rate (Decision Tree)', fontsize=14)
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.grid(True)
plt.xticks(max_depths)
plt.tight_layout()

plt.savefig('plot_error.png', dpi=800)
