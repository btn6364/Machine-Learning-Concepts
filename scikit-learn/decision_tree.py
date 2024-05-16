import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# Load dataset
pima = pd.read_csv("data/diabetes.csv", skiprows=1, names=col_names)
# print(pima.head())

# Split into features and labels 
features = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
X = pima[features]
y = pima.label

# Split the data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a decision tree 
dt = DecisionTreeClassifier() 

# Train the classifier 
dt = dt.fit(X_train, y_train)

# Use the trained tree to make predictions 
y_pred = dt.predict(X_test)

# print(y_pred)
print(f"Accuracy = {metrics.accuracy_score(y_test, y_pred)}")

# Visualize the decision tree
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())