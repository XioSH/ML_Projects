import pandas as pd
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import boxcox

# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)


# 3.1.1 数据的无量纲化处理
## 使用鸢尾花数据集中的数值特征来展示
Iris = pd.read_csv("03_特征工程/data/Iris.csv")
Iris2 = Iris.drop(['Id', "Species"], axis=1)
## (1) 标准化，并可视化标准化前后的数据变化情况
'''只减去均值
data_scale_1 = preprocessing.scale(Iris2, with_mean=True, with_std=False)
'''
'''减去均值后除以标准差
data_scale_2 = preprocessing.scale(Iris2, with_mean=True, with_std=True)
'''
'''另一种减去均值后除以标准差的方式
data_scale_3 = preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(Iris2)
'''
'''可视化原始数据和变换后的数据分布
labs = Iris2.columns.values
plt.figure(figsize=(16, 10))
plt.subplot(2, 2, 1)
plt.boxplot(Iris2.values, notch=True, labels=labs)
plt.grid()
plt.title('原始数据')
plt.subplot(2, 2, 2)
plt.boxplot(data_scale_1, notch=True, labels=labs)
plt.grid()
plt.title('只减去均值')
plt.subplot(2, 2, 3)
plt.boxplot(data_scale_2, notch=True, labels=labs)
plt.grid()
plt.title('减去均值后除以标准差')
plt.subplot(2, 2, 4)
plt.boxplot(data_scale_3, notch=True, labels=labs)      
plt.grid()
plt.title('减去均值后除以标准差（另一种方式）')
plt.subplots_adjust(wspace=0.1)
plt.show()
'''
## (2) 缩放，并可视化缩放前后的数据变化情况'''
'''min-max
data_minmax_1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(Iris2)
data_minmax_2 = preprocessing.MinMaxScaler(feature_range=(1, 10)).fit_transform(Iris2)
### 可视化数据缩放后的结果
labs = Iris2.columns.values
plt.figure(figsize=(25, 6))
plt.subplot(1, 3, 1)
plt.boxplot(Iris2.values, notch=True, labels=labs)
plt.grid()
plt.title('原始数据')   
plt.subplot(1, 3, 2)
plt.boxplot(data_minmax_1, notch=True, labels=labs)
plt.grid()
plt.title('缩放到[0, 1]区间')
plt.subplot(1, 3, 3)
plt.boxplot(data_minmax_2, notch=True, labels=labs)
plt.grid()
plt.title('缩放到[1, 10]区间')
plt.subplots_adjust(wspace=0.1)
plt.show()
'''
'''MaxAbsScaler
### 使训练集中每个特征的最大绝对值为1.0
data_maxabs = preprocessing.MaxAbsScaler().fit_transform(Iris2)
### 可视化数据缩放后的结果
labs = Iris2.columns.values 
plt.figure(figsize=(25, 6))
plt.subplot(1, 2, 1)
plt.boxplot(Iris2.values, notch=True, labels=labs)
plt.grid()
plt.title('原始数据')
plt.subplot(1, 2, 2)
plt.boxplot(data_maxabs, notch=True, labels=labs)
plt.grid()
plt.title('MaxAbsScaler')
plt.subplots_adjust(wspace=0.1)
plt.show()
'''
'''RobustScaler
### 对带有异常值的数据进行标准化
data_robs = preprocessing.RobustScaler(with_centering=True, with_scaling=True).fit_transform(Iris2)
data_stds = preprocessing.scale(Iris2, with_mean=True, with_std=True)
### 可视化数据缩放后的结果
labs = Iris2.columns.values     
plt.figure(figsize=(25, 6))

plt.subplot(1, 3, 1)
plt.boxplot(Iris2.values, notch=True, labels=labs)
plt.grid()
plt.title('原始数据')

plt.subplot(1, 3, 2)
plt.boxplot(data_stds, notch=True, labels=labs)
plt.grid()
plt.title('StandardScaler')

plt.subplot(1, 3, 3)
plt.boxplot(data_robs, notch=True, labels=labs)
plt.grid()
plt.title('RobustScaler')

plt.subplots_adjust(wspace=0.1)
plt.show()
'''
'''Normalizer
### 针对特征（列）
data_normL1_feature = preprocessing.normalize(Iris2, norm='l1', axis=0) 
data_normL2_feature = preprocessing.normalize(Iris2, norm='l2', axis=0)
### 针对样本（行）
data_normL1_sample = preprocessing.normalize(Iris2, norm='l1', axis=1)
data_normL2_sample = preprocessing.normalize(Iris2, norm='l2', axis=1)
### 可视化数据缩放后的结果
labs = Iris2.columns.values 
plt.figure(figsize=(15, 6))

plt.subplot(2, 2, 1)
plt.boxplot(data_normL1_feature, notch=True, labels=labs)
plt.grid()
plt.title('Normalizer L1 Feature')

plt.subplot(2, 2, 2)
plt.boxplot(data_normL2_feature, notch=True, labels=labs)
plt.grid()
plt.title('Normalizer L2 Feature')

plt.subplot(2, 2, 3)
plt.boxplot(data_normL1_sample, notch=True, labels=labs)
plt.grid()
plt.title('Normalizer L1 Sample')

plt.subplot(2, 2, 4)
plt.boxplot(data_normL2_sample, notch=True, labels=labs)
plt.grid()
plt.title('Normalizer L2 Sample')

plt.subplots_adjust(wspace=0.15)
plt.show()
'''

# 3.1.2 数据特征变换
'''对数变换
np.random.seed(12)
x = 1+np.random.poisson(lam=1.5, size=5000)+np.random.rand(5000)
lnx = np.log(x)
## 可视化变换前后的数据分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(x, bins=30, color='c', edgecolor='k', alpha=0.65)
plt.title('原始数据分布')       
plt.subplot(1, 2, 2)
plt.hist(lnx, bins=30, color='c', edgecolor='k', alpha=0.65)
plt.title('对数变换后数据分布') 
plt.subplots_adjust(wspace=0.2)
plt.show()
'''
'''Box-Cox变换
np.random.seed(12)
x = 1+np.random.poisson(lam=1.5, size=5000)+np.random.rand(5000)
bcx_1 = boxcox(x, lmbda=0)  # lmbda=0时，相当于对数变换
bcx_2 = boxcox(x, lmbda=0.5)
bcx_3 = boxcox(x, lmbda=2) 
bcx_4 = boxcox(x, lmbda=-1)
## 可视化变换前后的数据分布
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.hist(bcx_1, bins=50)
plt.title('$ln(x)$')

plt.subplot(2, 2, 2)
plt.hist(bcx_2, bins=50)
plt.title('$\sqrt{x}$')

plt.subplot(2, 2, 3)
plt.hist(bcx_3, bins=50)
plt.title('$x^2$')

plt.subplot(2, 2, 4)
plt.hist(bcx_4, bins=50)
plt.title('$ 1/x $')  

plt.subplots_adjust(hspace=0.4)
plt.show()
'''
'''正态变换'''
np.random.seed(12)
x = 1+np.random.poisson(lam=1.5, size=5000)+np.random.rand(5000)
QTn = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
QTnx = QTn.fit_transform(x.reshape(5000, 1))
## 可视化变换前后的数据分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)    
plt.hist(x, bins=50)
plt.title('原始数据分布')   
plt.subplot(1, 2, 2)
plt.hist(QTnx, bins=50)
plt.title('正态变换后数据分布')
plt.show()