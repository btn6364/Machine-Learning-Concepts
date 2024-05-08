import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as rnd 

# Create random 2D data
mu = np.array([10, 13])
sigma = np.array([[3.5, -1.8], [-1.8, 3.5]])

# print(f"mu shape = {mu.shape}")
# print(f"sigma shape = {sigma.shape}")

org_data = rnd.multivariate_normal(mu, sigma, size=(1000))
# print(f"Data Shape = {org_data.shape}")

plt.title("Original Data")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(org_data[:, 0], org_data[:, 1], color="red", s=10)
plt.show()

mean = np.mean(org_data, axis=0)
mean_data = org_data - mean
# print(f"Mean data = {mean_data.shape}")

# Compute the coviriance matrix 
cov = np.cov(mean_data.T) 
cov = np.round(cov, 2) 
# print(f"Cov shape = {cov.shape}")

# Get the eigen vectors and eigen values 
eig_val, eig_vec = np.linalg.eig(cov) 
# print(f"Eigen vectors = {eig_vec}")
# print(f"Eigen values = {eig_val}")

# Sort eigen values and corresponding eigen vectors in descending order
indices = np.arange(0,len(eig_val), 1)
indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
eig_val = eig_val[indices]
eig_vec = eig_vec[:,indices]
# print("Sorted Eigen vectors ", eig_vec)
# print("Sorted Eigen values ", eig_val, "\n")

# Get the explained variance
sum_eig_values = np.sum(eig_val)
explained_var = eig_val / sum_eig_values
# print(explained_var)
cum_var = np.cumsum(explained_var)
# print(cum_var)

# Project the data onto the eigen vectors (take the dot product) 
pca_data = np.dot(mean_data, eig_vec)
# print(pca_data)
# plt.title("PCA Data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.scatter(pca_data[:, 0], pca_data[:, 1], color="blue", s=10)
# plt.show()

# Reconstruct the data 
recons_X = np.dot(pca_data, eig_vec.T) + mean 

# Calculate the loss between reconstructed data and the original data 
loss = np.mean(np.square(recons_X - org_data))
print(f"Loss = {loss}")

plt.title("Reconstructed Data")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(recons_X[:, 0], recons_X[:, 1], color="blue", s=10)
plt.show()

