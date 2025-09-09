from imblearn.datasets import make_imbalance          
from imblearn.over_sampling import KMeansSMOTE,SMOTE,SVMSMOTE
from imblearn.under_sampling import AllKNN,CondensedNearestNeighbour,NearMiss
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 处理matplotlib中文显示和图像显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)
matplotlib.use('TkAgg')


# 准备不平衡数据
'''
# 对酒的特征数据进行标准化
wine_x,wine_y = load_wine(return_X_y=True)
wine_x = StandardScaler().fit_transform(wine_x)
## 将主成分分析提取的特征处理为类不平衡数据
pca = PCA(n_components = 13,random_state = 123)
pca.fit(wine_x)
pca_wine_x = pca.transform(wine_x)[:,0:3]
im_x,im_y = make_imbalance(pca_wine_x,wine_y,
                           sampling_strategy={0: 30, 1: 70, 2: 20},
                           random_state=12)
print(np.unique(im_y,return_counts = True))
'''

# 3.5.1 过采样方法
'''使用过采样算法KMeansSMOTE进行数据平衡
kmeans = KMeansSMOTE(random_state=123, k_neighbors=3)
kmeans_x, kmeans_y = kmeans.fit_resample(im_x,im_y)
print("KMeansSMOTE : ",np.unique(kmeans_y,return_counts = True))
'''
'''使用过采样算法SMOTE进行数据平衡
smote = SMOTE(random_state=123, k_neighbors=3)
smote_x, smote_y = smote.fit_resample(im_x,im_y)
print("SMOTE : ",np.unique(smote_y,return_counts = True))
'''
'''使用过采样算法SVMSMOTE进行数据平衡
svms = SVMSMOTE(random_state=123, k_neighbors=3)
svms_x, svms_y = svms.fit_resample(im_x,im_y)
print("SVMSMOTE : ",np.unique(svms_y,return_counts = True))
'''
'''可视化不同算法下的数据可视化结果，使用二维散点图
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(14,10))
## 原始数据分布
plt.subplot(2,2,1)
for ii,y in enumerate(im_y):
    plt.scatter(im_x[ii,0],im_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("不平衡数据")
## 过采样算法KMeansSMOTE
plt.subplot(2,2,2)
for ii,y in enumerate(kmeans_y):
    plt.scatter(kmeans_x[ii,0],kmeans_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("KMeansSMOTE")
## 过采样算法SMOTE
plt.subplot(2,2,3)
for ii,y in enumerate(smote_y):
    plt.scatter(smote_x[ii,0],smote_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("SMOTE")
## 过采样算法SVMSMOTE
plt.subplot(2,2,4)
for ii,y in enumerate(svms_y):
    plt.scatter(svms_x[ii,0],svms_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("SVMSMOTE")
plt.show()
'''

# 3.5.2 欠采样方法
'''使用欠采样算法CondensedNearestNeighbour进行数据平衡
cnn = CondensedNearestNeighbour(random_state=123, n_neighbors=7,n_seeds_S = 20)
cnn_x, cnn_y = cnn.fit_resample(im_x,im_y)
print("CondensedNearestNeighbour : ",np.unique(cnn_y,return_counts = True))
'''
'''使用欠采样算法AllKNN进行数据平衡
allknn = AllKNN(n_neighbors=10)
allknn_x, allknn_y = allknn.fit_resample(im_x,im_y)
print("AllKNN : ",np.unique(allknn_y,return_counts = True))
'''
'''使用欠采样算法NearMiss进行数据平衡
nmiss = NearMiss(n_neighbors=3)
nmiss_x, nmiss_y = nmiss.fit_resample(im_x,im_y)
print("NearMiss : ",np.unique(nmiss_y,return_counts = True))
'''
'''可视化不同算法下的数据可视化结果，使用二维散点图
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(14,10))
## 原始数据分布
plt.subplot(2,2,1)
for ii,y in enumerate(im_y):
    plt.scatter(im_x[ii,0],im_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("不平衡数据")
## 欠采样算法CondensedNearestNeighbour
plt.subplot(2,2,2)
for ii,y in enumerate(cnn_y):
    plt.scatter(cnn_x[ii,0],cnn_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("CondensedNearestNeighbour")
## 欠采样算法AllKNN
plt.subplot(2,2,3)
for ii,y in enumerate(allknn_y):
    plt.scatter(allknn_x[ii,0],allknn_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("AllKNN")
## 欠采样算法NearMiss
plt.subplot(2,2,4)
for ii,y in enumerate(nmiss_y):
    plt.scatter(nmiss_x[ii,0],nmiss_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("NearMiss")
plt.show()
'''

# 3.5.3 综合采样方法
'''使用过采样和欠采样的综合方法SMOTEENN进行数据平衡
smoteenn = SMOTEENN(random_state=123)
smoteenn_x, smoteenn_y = smoteenn.fit_resample(im_x,im_y)
print("SMOTEENN : ",np.unique(smoteenn_y,return_counts = True))
'''
'''使用过采样和欠采样的综合方法SMOTETomek进行数据平衡
smoteet = SMOTETomek(random_state=123)
smoteet_x, smoteet_y = smoteet.fit_resample(im_x,im_y)
print("SMOTETomek : ",np.unique(smoteet_y,return_counts = True))
'''
'''可视化不同算法下的数据可视化结果，使用二维散点图
colors = ["red","blue","green"]
shapes = ["o","s","*"]
fig = plt.figure(figsize=(12,5))
## 综合采样算法SMOTEENN
plt.subplot(1,2,1)
for ii,y in enumerate(smoteenn_y):
    plt.scatter(smoteenn_x[ii,0],smoteenn_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("SMOTEENN")
## 综合采样算法SMOTETomek
plt.subplot(1,2,2)
for ii,y in enumerate(smoteet_y):
    plt.scatter(smoteet_x[ii,0],smoteet_x[ii,1],s = 40,
                c = colors[y],marker = shapes[y])
    plt.title("SMOTETomek")
plt.show()
'''