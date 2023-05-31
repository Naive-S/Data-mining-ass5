import numpy as np

class DPC:
    def __init__(self, k=3):
        self.k = k

    def get_distance_matrix(self, datas):
        n = np.shape(datas)[0]
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                v_i = datas[i, :]
                v_j = datas[j, :]
                distance_matrix[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))
        return distance_matrix

    def select_dc(self, distance_matrix):
        n = np.shape(distance_matrix)[0]
        distance_array = np.reshape(distance_matrix, n * n)
        percent = 2.0 / 100
        position = int(n * (n - 1) * percent)
        dc = np.sort(distance_array)[position + n]
        return dc

    def get_local_density(self, distance_matrix, dc):
        n = np.shape(distance_matrix)[0]
        rhos = np.zeros(n)
        for i in range(n):
            rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
        return rhos

    def get_deltas(self, distance_matrix, rhos):
        n = np.shape(distance_matrix)[0]
        deltas = np.zeros(n)
        nearest_neighbor = np.zeros(n)
        rhos_index = np.argsort(-rhos)
        for i, index in enumerate(rhos_index):
            if i == 0:
                continue
            higher_rhos_index = rhos_index[:i]
            deltas[index] = np.min(distance_matrix[index, higher_rhos_index])
            nearest_neighbors_index = np.argmin(distance_matrix[index, higher_rhos_index])
            nearest_neighbor[index] = higher_rhos_index[nearest_neighbors_index].astype(int)
        deltas[rhos_index[0]] = np.max(deltas)
        return deltas, nearest_neighbor

    def find_k_centers(self, rhos, deltas):
        rho_and_delta = rhos * deltas
        centers = np.argsort(-rho_and_delta)
        return centers[:self.k]

    def density_peal_cluster(self, rhos, centers, nearest_neighbor):
        k = np.shape(centers)[0]
        if k == 0:
            print("Can't find any center")
            return
        n = np.shape(rhos)[0]
        labels = -1 * np.ones(n).astype(int)

        for i, center in enumerate(centers):
            labels[center] = i

        rhos_index = np.argsort(-rhos)
        for i, index in enumerate(rhos_index):
            if labels[index] == -1:
                labels[index] = labels[int(nearest_neighbor[index])]
        return labels

    def fit(self, datas):
        distance_matrix = self.get_distance_matrix(datas)
        dc = self.select_dc(distance_matrix)
        rhos = self.get_local_density(distance_matrix, dc)
        deltas, nearest_neighbor = self.get_deltas(distance_matrix, rhos)
        centers = self.find_k_centers(rhos, deltas)
        labels = self.density_peal_cluster(rhos, centers, nearest_neighbor)
        return labels

