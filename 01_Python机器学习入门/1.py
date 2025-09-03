from sklearn.linear_model import LinearRegression
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 图像中文显示的问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)

x = np.array([56, 32, 78, 160, 240, 89, 91, 69, 43])
y = np.array([90, 65, 125, 272, 312, 147, 159, 109, 78])

# 数据导入与处理，并进行数据探索
X=x.reshape(-1, 1)
Y=y.reshape(-1, 1)

#plt.figure(figsize=(10,6))
#plt.scatter(X, Y, s=50)
plt.title("原始数据的图")
#plt.show()

# 训练模型和预测
model = LinearRegression()
model.fit(X, Y)
x1 = np.array([40,]).reshape(-1, 1)
x1_pre = model.predict(np.array(x1))

# 数据可视化，将预测的点也打印在图上
plt.figure(figsize=(10, 8))
plt.scatter(X, Y)

b = model.intercept_
k = model.coef_
y = k * X + b
plt.plot(X, y)

y1 = k * x1 + b
plt.scatter(x1, y1,color='r')

plt.show()
