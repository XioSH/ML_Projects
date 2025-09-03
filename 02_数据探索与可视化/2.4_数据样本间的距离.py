import pandas as pd
from scipy.spatial import distance
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)

# 导入用于距离计算的数据
datadf = pd.read_csv("02_数据探索与可视化/data/种子数据.csv")
datadf2 = datadf.iloc[:,0:7]

# 计算样本距离并单独展示
'''欧式距离
dist = distance.cdist(datadf2, datadf2, "euclidean")
## 使用热力图可视化样本之间的距离
plt.figure(figsize=(8,6))
sns.heatmap(dist,cmap="YlGnBu")
plt.title("样本间欧式距离")
plt.show()
'''
'''曼哈顿距离
dist = distance.cdist(datadf2,datadf2,"cityblock")
## 使用热力图可视化样本之间的距离
plt.figure(figsize=(8,6))
sns.heatmap(dist,cmap="YlGnBu")
plt.title("样本间曼哈顿距离")
plt.show()
'''
'''切比雪夫距离
dist = distance.cdist(datadf2, datadf2, "chebyshev")
## 使用热力图可视化样本之间的距离
plt.figure(figsize=(8,6))
sns.heatmap(dist,cmap="YlGnBu")
plt.title("样本间切比雪夫距离")
plt.show()
'''
'''余弦距离
dist = distance.cdist(datadf2,datadf2,"cosine")
## 使用热力图可视化样本之间的距离
plt.figure(figsize=(8,6))
sns.heatmap(dist,cmap="YlGnBu")
plt.title("样本间余弦距离")
plt.show()
'''
'''相关系数距离
dist = distance.cdist(datadf2,datadf2,"correlation")
## 使用热力图可视化样本之间的距离
plt.figure(figsize=(8,6))
sns.heatmap(dist,cmap="YlGnBu")
plt.title("样本间相关系数距离")
plt.show()
'''
'''马氏距离
dist = distance.cdist(datadf2,datadf2,"mahalanobis")
## 使用热力图可视化样本之间的距离
plt.figure(figsize=(8,6))
sns.heatmap(dist,cmap="YlGnBu")
plt.title("样本间马氏距离")
plt.show()
'''

# 计算样本距离并集中展示
'''
# ---------------------- 1. 定义距离类型和对应的配置 ----------------------
# 用列表存储：(距离名称, 距离方法, 子图位置)，便于循环生成
distance_configs = [
    ("欧式距离", "euclidean", 1),    # 第1个子图（2行3列的第1个位置）
    ("曼哈顿距离", "cityblock", 2), # 第2个子图
    ("切比雪夫距离", "chebyshev", 3),# 第3个子图
    ("余弦距离", "cosine", 4),      # 第4个子图
    ("相关系数距离", "correlation", 5),# 第5个子图
    ("马氏距离", "mahalanobis", 6)  # 第6个子图
]

# ---------------------- 2. 创建统一画布和子图 ----------------------
# 设置总画布大小（宽18，高10，根据子图数量调整，确保每个子图不拥挤）
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
# 将 2x3 的 axes 数组展平为 1 维（方便循环索引）
axes = axes.flatten()

# ---------------------- 3. 循环生成每种距离的热力图 ----------------------
for name, metric, idx in distance_configs:
    # 计算当前距离矩阵
    if metric == "mahalanobis":
        # 马氏距离需要计算协方差矩阵的逆（注意：协方差矩阵需可逆）
        cov_matrix = np.cov(datadf2.T)  # 计算特征的协方差矩阵（转置是因为每行是样本）
        inv_cov_matrix = np.linalg.inv(cov_matrix)  # 协方差矩阵的逆
        dist = distance.cdist(datadf2, datadf2, metric=metric, VI=inv_cov_matrix)
    else:
        # 其他距离直接计算
        dist = distance.cdist(datadf2, datadf2, metric=metric)
    
    # 在对应子图上绘制热力图
    sns.heatmap(
        dist,
        cmap="YlGnBu",  # 统一颜色映射，确保对比一致性
        ax=axes[idx-1],  # 指定子图（idx从1开始，axes索引从0开始，需减1）
        cbar_kws={"shrink": 0.8}  # 调整颜色条大小，避免遮挡
    )
    
    # 设置子图标题（字体大小统一，增强美观）
    axes[idx-1].set_title(f"样本间{name}", fontsize=12, fontweight="bold")
    # 可选：隐藏x/y轴标签（样本索引通常无实际意义，隐藏后更简洁）
    axes[idx-1].set_xlabel("")
    axes[idx-1].set_ylabel("")

# ---------------------- 4. 调整布局和添加总标题 ----------------------
plt.suptitle("不同距离度量下的样本间距离热力图对比", fontsize=16, fontweight="bold", y=0.98)
# 自动调整子图间距（避免标题/标签重叠）
plt.tight_layout(rect=[0, 0, 1, 0.95])  # rect 为总标题预留空间（y从0到0.95）
# 显示图表
plt.show()
'''