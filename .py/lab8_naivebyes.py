import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

data = pd.read_csv("diabetes_data.csv")


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values


Y_binary = np.where(Y == 0, -1, 1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, test_size=0.3, random_state=45)


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)

nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
Y_score=nb.predict_proba(X_test)

svm_clf = svm.SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, Y_train)
Y_pred_svm = svm_clf.predict(X_test)


def evaluate_model(name, y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n{name} Evaluation:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("ROC AUC:", auc)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")


plt.figure(figsize=(8, 6))


evaluate_model("KNN", Y_test, Y_pred_knn, knn.predict_proba(X_test)[:, 1])
evaluate_model("Naive Bayes", Y_test, Y_pred_nb, nb.predict_proba(X_test)[:, 1])
evaluate_model("SVM", Y_test, Y_pred_svm, svm_clf.predict_proba(X_test)[:, 1])


plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc='lower right')
plt.grid()
plt.show()