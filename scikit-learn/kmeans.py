import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

home_data = pd.read_csv("data/housing.csv", usecols=["longitude", "latitude", "median_house_value"])
# print(home_data.head())
# sns.scatterplot(data = home_data, x = "longitude", y = "latitude", hue = "median_house_value")

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(home_data[["latitude", "longitude"]], home_data[["median_house_value"]], test_size=0.33, random_state=0)

# Normalize the datasets 
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# Fit the model 
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init = "auto")
kmeans.fit(X_train_norm)

# Visualize the results we just fit 
# sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
# plt.show()

# Draw a box plot
# sns.boxplot(x = kmeans.labels_, y = y_train['median_house_value'])
# plt.show()

score = silhouette_score(X_train_norm, kmeans.labels_, metric="euclidean")
# print(score)
inertia = kmeans.inertia_
print(inertia)


# Choose the best k
fits = []
inertias = [] 
scores = []
for k in range(2, 8): 
    model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_train_norm)
    fits.append(model) 
    inertias.append(model.inertia_)
    scores.append(silhouette_score(X_train_norm, model.labels_, metric="euclidean"))

# Draw using elbow method 
# sns.lineplot(x = range(2,8), y = inertias)
# plt.xlabel("K")
# plt.ylabel("Inertia")

sns.lineplot(x = range(2, 8), y = scores)
plt.show()