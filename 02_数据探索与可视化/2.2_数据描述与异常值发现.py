import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer

# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)

# 读取鸢尾花数据集
Iris = pd.read_csv('02_数据探索与可视化/data/Iris.csv')
Iris2 = Iris.drop(["Id", "Species"], axis=1)
#print(Iris2.head())

# 2.2.1 数据描述统计
'''(1) 数据集中程度的描述统计量
## 均值
print("均值：\n", Iris2.mean())
## 中位数
print("中位数：\n", Iris2.median())
## 众数
#print("众数：\n", Iris2.mode().iloc[0])
print("众数：\n", Iris2.mode())
'''
'''(2) 数据集中程度的描述统计量
## 极差
print("极差：\n", Iris2.max() - Iris2.min())
## 分位数
print("分位数：\n", Iris2.quantile(q=[0, 0.25, 0.5, 0.75, 1]))
## 方差
print("方差：\n", Iris2.var())
## 标准差
print("标准差：\n", Iris2.std())
## 变异系数
print("变异系数：\n", Iris2.std() / Iris2.mean())
'''
'''(3) 数据分布形态的描述统计量
## 偏度
print("偏度：\n", Iris2.skew())
## 峰度
print("峰度：\n", Iris2.kurt())
## 相关系数
iriscorr = Iris2.corr(method='pearson')
## 使用热力图进行可视化
plt.figure(figsize=(8, 6))
ax = sns.heatmap(iriscorr, fmt=".3f", annot=True, cmap='YlGnBu')
ax.set_yticklabels(iriscorr.index.values, va='center')
plt.title('Iris数据集各变量相关系数热力图', fontsize=16)
plt.show()
'''
'''(4) 单个数据变量的分布图
## 数量变量直方图可视化
plt.figure(figsize=(10, 6))
plt.hist(Iris2.PetalLengthCm, bins=30, color='lightblue', edgecolor='black')
plt.xlabel('PetalLengthCm')
plt.ylabel('频数')
plt.title('Iris数据集中PetalLengthCm变量的频数分布直方图')
plt.grid()
plt.show()
## 分类变量直方图可视化
plotdata = Iris.Species.value_counts()
plt.figure(figsize=(10, 6))
plt.bar(plotdata.index.values, plotdata.values, color='lightgreen', edgecolor='black')
plt.xlabel('数据种类')
plt.ylabel('频数')
plt.title('种类频数分布直方图')
plt.grid()
plt.show()
'''

# 2.2.2 发现异常值的基本方法
## KNN填充缺失值
oceandf = pd.read_csv("02_数据探索与可视化/data/热带大气海洋数据.csv")
knnimp = KNNImputer(n_neighbors=5)
oceandknn = knnimp.fit_transform(oceandf)
## 使用KNN填充缺失值后的oceandfknn数据中的5个变量演示
oceandfknn5 = pd.DataFrame(data=oceandknn[:, 3:8], columns=['SeaSurfaceTemp', 'AirTemp', 'Humidity', 'UWind', 'vWind'])
'''(1) 使用3sigma查看异常值
oceandfknn5_mean = oceandfknn5.mean()
oceandfknn5_std = oceandfknn5.std()
## 计算对应的样本是否为异常值
outlierindex = abs(oceandfknn5 - oceandfknn5_mean) > 3 * oceandfknn5_std
#print("各变量异常值情况：\n", outlierindex.sum())
'''
'''(2) 使用箱线图查看异常值
oceandfknn5.plot(kind='box', figsize=(10,6), title='使用KNN填充缺失值后的5个变量箱线图')
plt.grid()
plt.show()
'''
'''(3) 使用散点图查看异常值
x=[10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y=[7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')  
plt.show()
'''