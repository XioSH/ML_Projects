import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize,KBinsDiscretizer


# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)
matplotlib.use('TkAgg')

# 9.2.1 文本数据准备与可视化
## 读取数据，数据预处理，特征获取
bbcdf = pd.read_csv("09_贝叶斯算法和K-近邻算法/data/bbcdata.csv")

## 使用词云可视化不同类别数据的情况
'''
classification = np.unique(bbcdf.label)
plt.figure(figsize=(18,12))
for ii,cla in enumerate(classification):
    text = bbcdf.text_pre[bbcdf.label == cla]
    ## 设置词云参数
    WordC = WordCloud(margin=1,width=1000, height=1000,
                      max_words=200, min_font_size=10, 
                      background_color="white",max_font_size=200)
    WordC.generate_from_text(" ".join(text))
    plt.subplot(2,3,ii+1)
    plt.imshow(WordC)
    plt.title(cla,size = 40)
    plt.axis("off")
plt.tight_layout()
plt.show()   
'''

## 数据切分，训练集70%，测试集30%
X_train,X_test,y_train,y_test = train_test_split(bbcdf.text_pre,bbcdf.labelcode,test_size = 0.3, random_state=0)

## 获取数据的TF-IDF特征
vectorizer = CountVectorizer(stop_words="english",ngram_range=(1,2), max_features=4000)    
transformer = TfidfTransformer()
## 获取训练集的特征
train_tfidf = transformer.fit_transform(vectorizer.fit_transform(X_train))
train_tfidf = train_tfidf.toarray()
## 获取测试集的特征
test_tfidf = transformer.transform(vectorizer.transform(X_test))
test_tfidf = test_tfidf.toarray()


# 9.2.2 朴素贝叶斯文本分类
## 建立似然为高斯分布的朴素贝叶斯模型
gnb = GaussianNB().fit(train_tfidf, y_train)
gnb_pre = gnb.predict(test_tfidf)
###print(classification_report(y_test,gnb_pre))

## 建立似然为多项式分布的朴素贝叶斯模型
mnb = MultinomialNB().fit(train_tfidf, y_train)
mnb_pre = mnb.predict(test_tfidf)
###print(classification_report(y_test,mnb_pre))

## 建立似然为伯努利分布的朴素贝叶斯模型
bnb = BernoulliNB().fit(train_tfidf, y_train)
bnb_pre = bnb.predict(test_tfidf)
###print(classification_report(y_test,bnb_pre))

## 为方便后面可视化ROC曲线，对标签使用label_binarize进行编码
y_test_lb = label_binarize(y_test,classes=[0,1,2,3,4])
y_test_lb[0:5,:]
## 可视化三种算法的ROC曲线
model = [gnb,mnb,bnb]
modelname = ["GaussianNB","MultinomialNB","BernoulliNB"]
plt.figure(figsize=(15,5))
for ii,mod in enumerate(model):
    ## 对测试集进预测
    pre_score = mod.predict_proba(test_tfidf)
    ## 计算绘制ROC曲线的取值
    fpr_micro, tpr_micro, _ = roc_curve(y_test_lb.ravel(), pre_score.ravel())
    AUC = auc(fpr_micro, tpr_micro)  # AUC大小
    plt.subplot(1,3,ii+1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_micro, tpr_micro,"r",linewidth = 3)
    plt.xlabel("假正率")
    plt.ylabel("真正率")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.title(modelname[ii])
    plt.text(0.2,0.8,"AUC = "+str(round(AUC,4)))
plt.tight_layout()
plt.show()