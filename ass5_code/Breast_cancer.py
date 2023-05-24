# breast_cancer为WDBC数据集
# NMI、RI、purity、Silhouette Coefficient（四选二）
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans

breast = load_breast_cancer()
# print(breast.data.shape)
# print(breast.feature_names)
# print(breast.target)
data = breast.data
target = breast.target

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(data, target)
    pred = kmeans.predict(data)
    print(pred)
