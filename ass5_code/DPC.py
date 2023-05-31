import numpy as np

class DPC:
    def __init__(self, k=3):
        self.k = k

    def get_distance_matrix(self, datas):
        # 获取数据集中点的数量
        n = np.shape(datas)[0]
        # 构建一个 n x n 的全零矩阵，用于存储距离值
        distance_matrix = np.zeros((n, n))
        # 双重循环计算每对点之间的距离并将其存入距离矩阵中
        for i in range(n):
            for j in range(n):
                # 获取第 i 个点和第 j 个点的坐标向量
                v_i = datas[i, :]
                v_j = datas[j, :]
                # 计算欧氏距离并将其存入距离矩阵中
                distance_matrix[i, j] = np.sqrt(np.dot((v_i - v_j), (v_i - v_j)))
        # 返回距离矩阵
        return distance_matrix

    def select_dc(self, distance_matrix):
        # 获取距离矩阵的尺寸（假设距离矩阵为 n x n）
        n = np.shape(distance_matrix)[0]
        # 将距离矩阵转化为一维数组
        distance_array = np.reshape(distance_matrix, n * n)
        # 按照密度相对大小排序后取第二个百分之2位置处的值作为截断距离 DC
        percent = 2.0 / 100
        position = int(n * (n - 1) * percent)
        dc = np.sort(distance_array)[position + n]
        # 返回选定的截断距离 DC
        return dc

    def get_local_density(self, distance_matrix, dc):
        # 获取距离矩阵的尺寸（假设距离矩阵为 n x n）
        n = np.shape(distance_matrix)[0]
        # 创建一个长度为 n 的全零数组，用于存储每个点的局部密度
        rhos = np.zeros(n)
        # 遍历每个点，计算其在截断距离 DC 之内的相邻点的数量，并将该数量作为该点的局部密度
        for i in range(n):
            # 找出所有与第 i 个点距离小于截断距离 DC 的相邻点索引
            neighbors_idx = np.where(distance_matrix[i, :] < dc)[0]
            # 将该点本身从相邻点中删除，并将剩余点的数量作为该点的局部密度
            rhos[i] = neighbors_idx.shape[0] - 1
        # 返回每个点的局部密度
        return rhos

    def get_deltas(self, distance_matrix, rhos):
        # 获取距离矩阵的尺寸（假设距离矩阵为 n x n）
        n = np.shape(distance_matrix)[0]
        # 创建长度为 n 的全零数组，用于存储每个点的 Delta 值和最近高密度点的索引
        deltas = np.zeros(n)
        nearest_neighbor = np.zeros(n)
        # 对局部密度进行降序排列，并获取其排序后的索引
        rhos_index = np.argsort(-rhos)
        # 遍历所有点，计算其 Delta 值和最近高密度点的索引
        for i, index in enumerate(rhos_index):
            if i == 0:
                continue
            # 获取该点所有局部密度比它自己大的点的索引
            higher_rhos_index = rhos_index[:i]
            # 计算该点的 Delta 值，即到所有局部密度比它大的点中距离最近的点之间的距离
            deltas[index] = np.min(distance_matrix[index, higher_rhos_index])
            # 找到距离该点 Delta 值最小的那个点在高密度点集合中的索引
            nearest_neighbors_index = np.argmin(distance_matrix[index, higher_rhos_index])
            # 将该点的最近高密度点的索引存储到数组中
            nearest_neighbor[index] = higher_rhos_index[nearest_neighbors_index].astype(int)
        # 将局部密度最大的那个点的 Delta 值设为所有 Delta 值中的最大值
        deltas[rhos_index[0]] = np.max(deltas)
        # 返回每个点的 Delta 值和最近高密度点的索引
        return deltas, nearest_neighbor

    def find_k_centers(self, rhos, deltas):
        # 计算每个点的 rho x delta 值，并对其进行降序排列
        rho_and_delta = rhos * deltas
        centers = np.argsort(-rho_and_delta)
        # 返回前 k 个 rho x delta 值最大的点的索引，即为 k 个聚类中心的索引集合
        return centers[:self.k]

    def density_peal_cluster(self, rhos, centers, nearest_neighbor):
        # 获取聚类中心的数量 k
        k = np.shape(centers)[0]
        # 防止出现没有聚类中心的情况
        if k == 0:
            print("Can't find any center")
            return
        # 获取数据集中点的数量 n，并创建一个长度为 n 的全零数组，用于存储聚类标签
        n = np.shape(rhos)[0]
        labels = -1 * np.ones(n).astype(int)

        # 将每个聚类中心的标签设为其在 centers 数组中的索引
        for i, center in enumerate(centers):
            labels[center] = i

        # 根据 Delta 值从大到小的顺序遍历所有点，并根据其最近高密度点的标签来确定自己的标签
        rhos_index = np.argsort(-rhos)
        for i, index in enumerate(rhos_index):
            if labels[index] == -1:
                # 如果该点没有被聚类中心覆盖，则将其标签设为其最近高密度点的标签
                labels[index] = labels[int(nearest_neighbor[index])]
        # 返回所有点的聚类标签
        return labels

    def fit(self, datas):
        # 计算距离矩阵
        distance_matrix = self.get_distance_matrix(datas)
        # 选择截断距离
        dc = self.select_dc(distance_matrix)
        # 计算局部密度
        rhos = self.get_local_density(distance_matrix, dc)
        # 计算delta和最近邻
        deltas, nearest_neighbor = self.get_deltas(distance_matrix, rhos)
        # 寻找k个中心点
        centers = self.find_k_centers(rhos, deltas)
        # 密度峰聚类
        labels = self.density_peal_cluster(rhos, centers, nearest_neighbor)
        return labels
