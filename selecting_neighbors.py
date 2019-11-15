# Author: Adrien Bibal
# Select neighbors that will be used for explaining locally an embedding

import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.neighbors import NearestNeighbors

import utils

def find_k_neighbors(embedding, i, k):
	"""
	From an embedding, returns the ID of the k nearest neighbors of a certain point at index i. 
	"""

	# We look for k+1 neighbors because i is considered as a neoighbor of itself.
	_, IDs =  NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embedding).kneighbors(embedding)
	neighbor_ID = IDs[i]

	# Remove i from IDs
	neighbor_ID = np.delete(neighbor_ID, np.where(neighbor_ID == i))

	return neighbor_ID

def store_neighbors_dataset(embedding, dataset, neighbor_IDs):
	"""
	Create new files with neighbor_IDs as a subset of embedding and dataset instances.
	"""

	neighbors_embedding = embedding.iloc[neighbor_IDs,]
	neighbors_dataset   = dataset.iloc[neighbor_IDs,]

	neighbors_embedding.to_csv("dataset/embedding.csv", header=None, index=None)
	neighbors_dataset.to_csv("dataset/dataset.csv", index=None)
	pd.DataFrame(neighbor_IDs).to_csv("dataset/neighbor_IDs.csv", header=None, index=None)

def show_embedding(embedding, instance_names):
	utils.scatter_with_samples(Y=np.array(embedding), texts=instance_names)


if __name__ == "__main__":
	embedding      = pd.read_csv("dataset/country_embedding.csv", header=None)
	dataset        = pd.read_csv("dataset/country_dataset.csv")
	instance_names = pd.read_csv("dataset/country_names.csv", header=None)
	instance_names = np.array(instance_names.iloc[:,0])

	neighbor_IDs = find_k_neighbors(embedding, 104, 10)
	print("The instance that is queried is", instance_names[104])
	print("Its neighbors are", instance_names[neighbor_IDs])

	# show_embedding(embedding, instance_names)

	store_neighbors_dataset(embedding, dataset, neighbor_IDs)
	