from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)
matplotlib.use('TkAgg')

# 对酒的特征数据进行标准化
wine_x,wine_y = load_wine(return_X_y=True)
wine_x = StandardScaler().fit_transform(wine_x)

# 3.4.1 主成分分析法PCA
'''
## 使用主成分分析对酒数据集进行降维
pca = PCA(n_components = 13,random_state = 123)
pca.fit(wine_x)
## 可视化主成分分析的解释方差得分
exvar = pca.explained_variance_
plt.figure(figsize=(10,6))
plt.plot(exvar,"r-o")
plt.hlines(y = 1, xmin = 0, xmax = 12)
plt.xlabel("特征数量")
plt.ylabel("解释方差大小")
plt.title("主成分分析")
plt.show()

## 可以发现使用数据的前3个主成分较合适
pca_wine_x = pca.transform(wine_x)[:,0:3]
print(pca_wine_x.shape)

## 在3D空间中可视化主成分分析后的数据空间分布
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(10,6))
## 将坐标系设置为3D
ax1 = fig.add_subplot(111, projection="3d")
for ii,y in enumerate(wine_y):
    ax1.scatter(pca_wine_x[ii,0],pca_wine_x[ii,1],pca_wine_x[ii,2],
                s = 40,c = colors[y],marker = shapes[y])
ax1.set_xlabel("主成分1",rotation=20)
ax1.set_ylabel("主成分2",rotation=-20)
ax1.set_zlabel("主成分3",rotation=90)
ax1.azim = 225
ax1.set_title("主成分特征空间可视化")
plt.show()
'''

# 3.4.2 核主成分分析法KernelPCA
'''
## 使用核主成分分析获取数据的主成分
kpca = KernelPCA(n_components = 13,kernel = "rbf", ## 核函数为rbf核
                 gamma = 0.2,random_state = 123)
kpca.fit(wine_x)
## 可视化核主成分分析的中心矩阵特征值
eigenvalues = kpca.eigenvalues_
plt.figure(figsize=(10,6))
plt.plot(eigenvalues,"r-o")
plt.hlines(y = 4, xmin = 0, xmax = 12)
plt.xlabel("特征数量")
plt.ylabel("中心核矩阵的特征值大小")
plt.title("核主成分分析")
plt.show()
## 获取前3个核主成分
kpca_wine_x = kpca.transform(wine_x)[:,0:3]
print(kpca_wine_x.shape)
## 在3D空间中可视化主成分分析后的数据空间分布
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(10,6))
## 将坐标系设置为3D
ax1 = fig.add_subplot(111, projection="3d")
for ii,y in enumerate(wine_y):
    ax1.scatter(kpca_wine_x[ii,0],kpca_wine_x[ii,1],kpca_wine_x[ii,2],
                s = 40,c = colors[y],marker = shapes[y])
ax1.set_xlabel("核主成分1",rotation=20)
ax1.set_ylabel("核主成分2",rotation=-20)
ax1.set_zlabel("核主成分3",rotation=90)
ax1.azim = 225
ax1.set_title("核主成分特征空间可视化")
plt.show()
'''

# 3.4.3 流形学习方法
'''
## 流形学习进行数据的非线性降维
isomap = Isomap(n_neighbors = 7,## 每个点考虑的近邻数量
                n_components = 3) ## 降维到3维空间中
## 获取降维后的数据
isomap_wine_x = isomap.fit_transform(wine_x)
print(isomap_wine_x.shape)
## 在3D空间中可视化流行降维后的数据空间分布
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(10,6))
## 将坐标系设置为3D
ax1 = fig.add_subplot(111, projection="3d")
for ii,y in enumerate(wine_y):
    ax1.scatter(isomap_wine_x[ii,0],isomap_wine_x[ii,1],isomap_wine_x[ii,2],
                s = 40,c = colors[y],marker = shapes[y])
ax1.set_xlabel("特征1",rotation=20)
ax1.set_ylabel("特征2",rotation=-20)
ax1.set_zlabel("特征3",rotation=90)
ax1.azim = 225
ax1.set_title("Isomap降维可视化")
plt.show()
'''

# 3.4.4 t-SNE方法
'''
## TSNE进行数据的降维,降维到3维空间中
tsne = TSNE(n_components = 3,perplexity =25,
            early_exaggeration =3,random_state=123) 
## 获取降维后的数据
tsne_wine_x = tsne.fit_transform(wine_x)
print(tsne_wine_x.shape)
## 在3D空间中可视化流行降维后的数据空间分布
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(10,6))
## 将坐标系设置为3D
ax1 = fig.add_subplot(111, projection="3d")
for ii,y in enumerate(wine_y):
    ax1.scatter(tsne_wine_x[ii,0],tsne_wine_x[ii,1],tsne_wine_x[ii,2],
                s = 40,c = colors[y],marker = shapes[y])
ax1.set_xlabel("特征1",rotation=20)
ax1.set_ylabel("特征2",rotation=-20)
ax1.set_zlabel("特征3",rotation=90)
ax1.azim = 225
ax1.set_title("TSNE降维可视化")
plt.show()
'''

# 3.4.5 多维尺度分析MDS
'''
## MDS进行数据的降维,降维到3维空间中
mds = MDS(n_components = 3,dissimilarity = "euclidean",random_state=123) 
## 获取降维后的数据
mds_wine_x = mds.fit_transform(wine_x)
print(mds_wine_x.shape)
## 在3D空间中可视化流行降维后的数据空间分布
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(10,6))
## 将坐标系设置为3D
ax1 = fig.add_subplot(111, projection="3d")
for ii,y in enumerate(wine_y):
    ax1.scatter(mds_wine_x[ii,0],mds_wine_x[ii,1],mds_wine_x[ii,2],
                s = 40,c = colors[y],marker = shapes[y])
ax1.set_xlabel("特征1",rotation=20)
ax1.set_ylabel("特征2",rotation=-20)
ax1.set_zlabel("特征3",rotation=90)
ax1.azim = 225
ax1.set_title("MDS降维可视化")
plt.show()
'''