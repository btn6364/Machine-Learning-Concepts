from sklearn.datasets import load_breast_cancer
import numpy as np 
import pandas as pd
from keras.datasets import cifar10
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

breast = load_breast_cancer()
breast_data = breast.data
# print(f"Shape = {breast_data.shape}")

labels = np.reshape(breast.target, (569,1))

final_breast_data = np.concatenate([breast_data, labels], axis=1)
# print(final_breast_data.shape)

breast_dataset = pd.DataFrame(final_breast_data)
# print(breast_dataset)

features = breast.feature_names
features_labels = np.append(features, "label")
breast_dataset.columns = features_labels

# print(breast_dataset.head())

breast_dataset["label"].replace(0, "Benign", inplace=True)
breast_dataset["label"].replace(1, "Malignant", inplace=True)

# print(breast_dataset.tail())

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('Traning data shape:', x_train.shape)
# print('Testing data shape:', x_test.shape)

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
# print('Total number of outputs : ', nClasses)
# print('Output classes : ', classes)

# Normalize the data
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) 

# print(f"Mean after normalization = {np.mean(x)}")
# print(f"Standard deviation after normalization = {np.std(x)}")

# Convert the normalized data into a DataFrame 
feature_columns = [f"feature{i}" for i in range(x.shape[1])]
normalized_breast = pd.DataFrame(x, columns=feature_columns)
# print(normalized_breast.tail())

pca_breast = PCA(n_components=2)
prinple_components_breast = pca_breast.fit_transform(x) 
# print(prinple_components_breast)

pca_breast_df = pd.DataFrame(data = prinple_components_breast, columns=["PC1", "PC2"])
# print(pca_breast_df.tail())

# How much information each PC is holding?
print(pca_breast.explained_variance_ratio_)

# Plot the graph
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(pca_breast_df.loc[indicesToKeep, 'PC1']
               , pca_breast_df.loc[indicesToKeep, 'PC2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
# plt.show()

# Visualiza Cifar-10
x_train = x_train / 255.0
x_train_flat = x_train.reshape(-1, 3072) 
feature_columns = [f"pixel{i}" for i in range(x_train_flat.shape[1])]
cifar_df = pd.DataFrame(x_train_flat, columns=feature_columns)
cifar_df["label"] = y_train

# print(cifar_df.head())

pca_cifar = PCA(n_components=2)
principle_component_cifar = pca_cifar.fit_transform(cifar_df.iloc[:,:-1])
pca_cifar_df = pd.DataFrame(data = principle_component_cifar, columns = ["PC1", "PC2"])
print(pca_cifar_df.head())