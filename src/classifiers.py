from collections import Counter

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Bohdan
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

#load train and test datasets
train_df = pd.read_csv('../dataset/train.csv')
test_df = pd.read_csv('../dataset/test.csv')
y_test_df = pd.read_csv('../dataset/gender_submission.csv')

#preprocess the data
def preprocess_data(df):
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

#extract target variable
X_train = train_df.drop('Survived', axis=1).values
y_train = train_df['Survived'].values

X_test = test_df.values
y_test = y_test_df['Survived'].values

#initialize classifiers
decision_tree = DecisionTreeClassifier()
svm_classifier = SVC(probability=True)
logistic_regression = LogisticRegression(max_iter=1000)
knn_classifier = KNeighborsClassifier()
knn_custom = KNN(k=5)

#Mikhail
#train classifiers
decision_tree.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
logistic_regression.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
knn_custom.fit(X_train, y_train)

#make predictions
y_pred_decision_tree = decision_tree.predict(X_test)
y_pred_svm = svm_classifier.predict(X_test)
y_pred_logistic_regression = logistic_regression.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_knn_custom = knn_custom.predict(X_test)

# evaluate classifiers
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_decision_tree))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logistic_regression))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Custom KNN Accuracy:", accuracy_score(y_test, y_pred_knn_custom))

# Calculate ROC curve and AUC for each model
classifiers = [decision_tree, svm_classifier, logistic_regression, knn_classifier, knn_custom]
classifier_names = ['Decision Tree', 'SVM', 'Logistic Regression', 'KNN', 'Custom KNN']
predictions = [y_pred_decision_tree, y_pred_svm, y_pred_logistic_regression, y_pred_knn, y_pred_knn_custom]

plt.figure(figsize=(10, 10))

for clf, y_pred, name in zip(classifiers, predictions, classifier_names):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
