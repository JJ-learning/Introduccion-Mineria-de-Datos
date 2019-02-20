import numpy as np
import pandas as pd
import sklearn

from sklearn.cluster import KMeans

def clustering(atributes, amount_centroides):
    centroides = atributes[np.random.choice(atributes.shape[0], amount_centroides, replace=False)]
    kmeans = KMeans(n_clusters = amount_centroides, init = centroides, max_iter = 500, n_init = 1, random_state = 0)
    distances = kmeans.fit_transform(atributes)
    centros = kmeans.cluster_centers_

    return kmeans, distances, centros

def readData(fichero):
    data = pd.read_csv(fichero, header=None)
    atributes = data.values[1:, 0:-1]
    classes = data.values[1:, -1]

    return atributes, classes

def calculateSSE(distances):
    SSE = 0
    for i in range(distances.shape[0]):
        SSE += pow(distances[i], 2)

    return SSE

if __name__ == '__main__':
    fichero = "../Datasets/iris.arff"
    amount_centroides = 5
    seed = 15

    np.random.seed(seed)
    atributes, classes = readData(fichero)
    kmeans, distances, centros = clustering(atributes, amount_centroides)
    SSE = calculateSSE(distances)

    print("The centroids are: ", centros)
    print("The SSE is: ", SSE)
