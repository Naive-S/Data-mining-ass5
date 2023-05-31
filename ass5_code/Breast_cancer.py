# breast_cancer为WDBC数据集
# NMI、RI、purity、Silhouette Coefficient（四选二）
# K-Means、DBSCAN、SpectralClustering（谱聚类）、EM算法（高斯混合模型）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, rand_score

breast_cancer = load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

nmi_list = []
ri_list = []
modelname = ['Kmeans','DBSCAN', 'SpectralClustering', 'EM']


class modelsec:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def modelselection(self):
        print('{}'.format(self.model))
        if self.model == 'kmeans':
            labels = kmeans.labels_
            silhouette_coefficient = silhouette_score(self.X, labels)
            print("Silhouette Coefficient: ", silhouette_coefficient)
        elif self.model == 'DBSCAN':
            labels = DBSCAN.fit_predict(self.X)
        elif self.model == 'spectral':
            labels = spectral.fit_predict(self.X)
            silhouette_coefficient = silhouette_score(self.X, labels)
            print("Silhouette Coefficient: ", silhouette_coefficient)
        elif self.model == 'gmm':
            labels = gmm.fit_predict(data)
            silhouette_coefficient = silhouette_score(self.X, labels)
            print("Silhouette Coefficient: ", silhouette_coefficient)
        nmi = normalized_mutual_info_score(self.y, labels)
        ri = rand_score(self.y, labels)
        nmi_list.append(nmi)
        ri_list.append(ri)
        print("NMI: ", nmi)
        print("RI: ", ri)
        return labels

    def paint(self, labels):
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.title('{}'.format(self.model))
        plt.show()


# 构建模型、训练模型
kmeans = KMeans(n_clusters=2, n_init="auto").fit(data)
DBSCAN = DBSCAN(eps=0.7, min_samples=5).fit(data)
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors').fit(data)
gmm = GaussianMixture(n_components=2, random_state=0).fit(data)
# kmeans
k = modelsec(data, target, 'kmeans')
k.paint(k.modelselection())

# DBSCAN
d = modelsec(data, target, 'DBSCAN')
d.paint(d.modelselection())

# 谱聚类
s = modelsec(data, target, 'spectral')
s.paint(s.modelselection())

# EM高斯模型
g = modelsec(data, target, 'gmm')
g.paint(g.modelselection())

############################## 绘图比较 ####################################
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# 设置位置和宽度
x_pos = np.arange(len(modelname))
bar_width = 0.35

# 绘制簇形柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(x_pos - bar_width / 2, nmi_list, bar_width, label='NMI')
rects2 = ax.bar(x_pos + bar_width / 2, ri_list, bar_width, label='RI')

# 添加标签和标题
ax.set_xticks(x_pos)
ax.set_xticklabels(modelname)
ax.set_ylabel('得分')
ax.set_title('（乳腺癌）不同聚类算法的NMI和RI得分比较')
ax.legend()

plt.bar_label(rects1)
plt.bar_label(rects2)
plt.show()
