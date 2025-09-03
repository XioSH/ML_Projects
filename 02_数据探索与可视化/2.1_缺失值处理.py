# 模块导入
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import missingno as msno
import altair as alt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
#from missingpy import MissForest

# 图像中文显示的问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)

# 读取用于演示的数据集
oceandf = pd.read_csv("02_数据探索与可视化/data/热带大气海洋数据.csv")

# 2.1.1 简单的缺失值处理方法
'''(1) 发现数据中的缺失值
#### 判断每个变量中是否存在缺失值
pd.isna(oceandf).sum()
#### 使用可视化方法查看缺失值在数据中的分布
msno.matrix(oceandf, figsize=(14, 7), width_ratios=(13, 2), color=(0.25, 0.25, 0.5))
plt.show()
'''
'''(2) 剔除带有缺失值的行或列
#### 删除带有缺失值的行
oceandf2 = oceandf.dropna(axis=0)
oceandf2.info()
#### 删除带有缺失值的列
oceandf3 = oceandf.dropna(axis=1)
oceandf3.info()
'''
'''(3) 对缺失值进行插补
#### 可视化出剔除缺失值所在行后AirTemp和Humidity变量的数据分布散点图
plt.figure(figsize=(10, 6))
plt.grid()
plt.xlabel('AirTemp')
plt.ylabel('Humidity')
plt.title('剔除带有缺失值的行')
plt.show()

#### 找到缺失值所在的位置
nanaindex = pd.isna(oceandf.AirTemp) | pd.isna(oceandf.Humidity)

#### 使用缺失值前面的值进行填充
oceandf4 = oceandf.fillna(axis=0, method='ffill')
#### 可视化填充后的结果
plt.figure(figsize=(10, 6))
plt.scatter(oceandf4.AirTemp[~nanaindex], oceandf4.Humidity[~nanaindex], c='blue', marker='o', label='非缺失值')
plt.scatter(oceandf4.AirTemp[nanaindex], oceandf4.Humidity[nanaindex], c='red', marker='s', label='缺失值')
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('AirTemp')
plt.ylabel('Humidity')  
plt.title('使用缺失值前面的值进行填充')
plt.show()

#### 使用缺失值后面的值进行填充
oceandf4 = oceandf.fillna(axis=0, method='bfill')
#### 可视化填充后的结果
plt.figure(figsize=(10, 6))
plt.scatter(oceandf4.AirTemp[~nanaindex], oceandf4.Humidity[~nanaindex], c='blue', marker='o', label='非缺失值')
plt.scatter(oceandf4.AirTemp[nanaindex], oceandf4.Humidity[nanaindex], c='red', marker='s', label='缺失值')
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('AirTemp')
plt.ylabel('Humidity')  
plt.title('使用缺失值后面的值进行填充')
plt.show()

#### 使用变量均值进行填充
AirTempmean = oceandf.AirTemp.mean()
Humiditymean = oceandf.Humidity.mean()
AirTemp = oceandf.AirTemp.fillna(value=AirTempmean)
Humidity = oceandf.Humidity.fillna(value=Humiditymean)
#### 可视化填充后的结果
plt.figure(figsize=(10, 6))
plt.scatter(AirTemp[~nanaindex], Humidity[~nanaindex], c='blue', marker='o', label='非缺失值')
plt.scatter(AirTemp[nanaindex], Humidity[nanaindex], c='red', marker='s', label='缺失值')
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('AirTemp')
plt.ylabel('Humidity')  
plt.title('使用变量均值填充')
plt.show()
'''

# 2.1.2 复杂的缺失值填充方法
''' (1) IterativeImputer多变量缺失值填充
nanaindex = pd.isna(oceandf.AirTemp) | pd.isna(oceandf.Humidity)
iterimp = IterativeImputer(random_state=123)
oceandfiter = iterimp.fit_transform(oceandf)
#### 获取填充后的变量
AirTemp = oceandfiter[:, 4]
Humidity = oceandfiter[:, 5]
#### 可视化填充后的结果
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
plt.scatter(AirTemp[~nanaindex], Humidity[~nanaindex], c='blue', marker='o', label='非缺失值')
plt.scatter(AirTemp[nanaindex], Humidity[nanaindex], c='red', marker='s', label='缺失值')
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('AirTemp')
plt.ylabel('Humidity')  
plt.title('使用IterativeImputer方法填充')
plt.show()
'''
''' (2) K-近邻缺失值填充
nanaindex = pd.isna(oceandf.AirTemp) | pd.isna(oceandf.Humidity)
knnimp = KNNImputer(n_neighbors=5)
oceandknn = knnimp.fit_transform(oceandf)
#### 获取填充后的变量
AirTemp = oceandknn[:, 4]
Humidity = oceandknn[:, 5]
#### 可视化填充后的结果
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
plt.scatter(AirTemp[~nanaindex], Humidity[~nanaindex], c='blue', marker='o', label='非缺失值')
plt.scatter(AirTemp[nanaindex], Humidity[nanaindex], c='red', marker='s', label='缺失值')
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('AirTemp')
plt.ylabel('Humidity')  
plt.title('使用KNNImputer方法填充')
plt.show()
'''
''' (3) 随机森林缺失值填充
nanaindex = pd.isna(oceandf.AirTemp) | pd.isna(oceandf.Humidity)
forestimp = MissForest(n_estimators=100, random_state=123)
oceandknn = forestimp.fit_transform(oceandf)
#### 获取填充后的变量
AirTemp = oceandknn[:, 4]
Humidity = oceandknn[:, 5]
#### 可视化填充后的结果
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
plt.scatter(AirTemp[~nanaindex], Humidity[~nanaindex], c='blue', marker='o', label='非缺失值')
plt.scatter(AirTemp[nanaindex], Humidity[nanaindex], c='red', marker='s', label='缺失值')
plt.grid()
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('AirTemp')
plt.ylabel('Humidity')  
plt.title('使用MissForest方法填充')
plt.show()
'''