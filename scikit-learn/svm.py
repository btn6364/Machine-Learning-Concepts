from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

cancer = datasets.load_breast_cancer() 

# print(f"Features = {cancer.feature_names}")
# print(f"Labels = {cancer.target_names}")

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=42)

classifier = svm.SVC(kernel="linear")

# Train the model 
classifier.fit(X_train, y_train)

# Make prediction
y_pred = classifier.predict(X_test)

# Evaluate our model 
from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# print(confusion_matrix)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy Score = {accuracy}")