import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)
matplotlib.use('TkAgg')

# 3.2.1 分类特征重新编码
## 准备类别标签数据
np.random.seed(12)
Iris = pd.read_csv("03_特征工程/data/Iris.csv")
label = np.random.choice(Iris.Species.values, size=4, replace=False)
label = label.reshape(-1, 1)
'''OrdinalEncoder编码为常数
OrdE = preprocessing.OrdinalEncoder()
label_OrdE = OrdE.fit_transform(label)
print("分类特征编码为常数：\n", label_OrdE)
'''
'''LabelEncoder编码为0~n-1的整数
le = preprocessing.LabelEncoder()
label_le = le.fit_transform([1,2,3,10,10])
print("分类特征编码为0~n-1的整数：\n", label_le)
'''
'''OneHotEncoder独热编码
OneHotE = preprocessing.OneHotEncoder()
label_OneHotE = OneHotE.fit_transform(label)
print("独热编码结果：\n", label_OneHotE.toarray())
'''
'''LabelBinarizer二值化
LB = preprocessing.LabelBinarizer()
label_LB = LB.fit_transform(label)
print("二值化结果：\n", label_LB)
'''
'''MultiLabelBinarizer多标签二值化
mlb = preprocessing.MultiLabelBinarizer()
label_mlb = mlb.fit_transform([('A','B'), ('B','C'), ('D')])
print("多标签二值化结果：\n", label_mlb)  
'''  

# 3.2.2 数值特征重新编码
'''一个变量的多项式特征
X = np.arange(1,5).reshape(-1,1)
polyF = preprocessing.PolynomialFeatures(degree=3, include_bias=False)
X_polyF = polyF.fit_transform(X)
print("多项式特征结果：\n", X_polyF)
'''
'''多个变量的多项式特征
X = np.arange(1,11).reshape(-1,2)
polyF = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
X_polyF = polyF.fit_transform(X)
print("多项式特征结果：\n", X_polyF)
'''
'''分箱操作的编码
X = Iris.iloc[:, 1:5].values
n_bin = [2,3,4,5]
Kbins = preprocessing.KBinsDiscretizer(n_bins=n_bin, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf')
X_Kbins = Kbins.fit_transform(X)
## 获取划分区间时的分界线
X_Kbins_edges = Kbins.bin_edges_
## 对分箱前后的数据进行可视化
plt.figure(figsize=(16, 8))
## 可视化分箱前的特征
for ii in range(X.shape[1]):
    plt.subplot(2, 4, ii+1)
    plt.hist(X[:, ii], bins=30, color='steelblue', edgecolor='black')
    plt.title(Iris.columns[ii+1])
    ## 可视化分箱的分界线
    edges = X_Kbins_edges[ii]
    for edge in edges:
        plt.vlines(edge, 0, 25, color='red', linestyle='--', linewidth=3)
## 可视化分箱后的特征
for ii, binsii in enumerate(n_bin):
    plt.subplot(2, 4, ii+5)
    ## 计算每个元素出现的次数
    barx, height = np.unique(X_Kbins[:, ii], return_counts=True)
    plt.bar(barx, height)
plt.show()
'''

# 3.2.3 文本数据的特征构建
## 读取一个文本文件
textdf = pd.read_table("03_特征工程/data/文本数据.txt", header=0)
## 将所有大写字母转化为小写
textdf['text'] = textdf.text.apply(lambda x: x.lower())
## 剔除多余的空格和标点符号
textdf['text'] = textdf.text.apply(lambda x: re.sub(r'[^\w\s]', '', x))
'''统计词频
text = ' '.join(textdf.text)
text = text.split(' ')
## 计算每个词出现的次数
textfre = pd.Series(text).value_counts()
## 使用条形图可视化词频
textfre.plot(kind='bar', figsize=(10, 6), rot=90)
plt.xlabel('单词')
plt.ylabel('词频')
plt.title('文本数据词频统计')
plt.show()
'''
'''词袋模型
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(textdf.text)
cv_matrix_df = pd.DataFrame(cv_matrix.toarray(), columns=cv.get_feature_names_out())
## 通过余弦相似性计算文本之间的相关系数
textcosin = cosine_similarity(cv_matrix_df)
plt.figure(figsize=(8, 6))
sns.heatmap(textcosin, fmt='.2f', annot=True, cmap='YlGnBu')    
plt.title('文本TF特征余弦相似性热力图')
plt.show()
## 获取文本的TF-IDF特征
TFIDF = TfidfVectorizer(stop_words='english')
TFIDF_matrix = TFIDF.fit_transform(textdf.text).toarray()
print(TFIDF_matrix)
textcosin_tfidf = cosine_similarity(TFIDF_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(textcosin_tfidf, fmt='.2f', annot=True, cmap='YlGnBu')    
plt.title('文本TF-IDF特征余弦相似性热力图')
plt.show()
'''