import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
data = pd.read_csv('diabetes_data.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into training (first 600) and test (last 10)
X_train, X_test = X[:600], X[-10:]
y_train, y_test = y[:600], y[-10:]

# Create three weak learners with different sample ranges
learner1 = DecisionTreeClassifier()  # Samples 1-200
learner2 = SVC(probability=True)  # Samples 201-400
learner3 = KNeighborsClassifier(n_neighbors=3)  # Samples 401-600

# Train each learner on their specific sample range
learner1.fit(X_train[:200], y_train[:200])
learner2.fit(X_train[200:400], y_train[200:400])
learner3.fit(X_train[400:600], y_train[400:600])


# Make predictions on test data using bagging (majority voting)
def bagging_predict(X):
    pred1 = learner1.predict(X)
    pred2 = learner2.predict(X)
    pred3 = learner3.predict(X)

    # Combine predictions using majority vote
    final_pred = []
    for p1, p2, p3 in zip(pred1, pred2, pred3):
        votes = {0: 0, 1: 0}
        votes[p1] += 1
        votes[p2] += 1
        votes[p3] += 1
        final_pred.append(max(votes.items(), key=lambda x: x[1])[0])
    return final_pred


# Test on the last 10 samples
y_pred = bagging_predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Print individual predictions and final decision
print("\nDetailed predictions for test samples:")
for i, (true, pred) in enumerate(zip(y_test, y_pred)):
    p1 = learner1.predict([X_test[i]])[0]
    p2 = learner2.predict([X_test[i]])[0]
    p3 = learner3.predict([X_test[i]])[0]
    print(f"Test sample {i + 1}:")
    print(f"  Learner1 (Decision Tree) = {p1}")
    print(f"  Learner2 (SVM) = {p2}")
    print(f"  Learner3 (KNN) = {p3}")
    print(f"  Overall decision = {pred}")
    print(f"  True value = {true}")
    print()