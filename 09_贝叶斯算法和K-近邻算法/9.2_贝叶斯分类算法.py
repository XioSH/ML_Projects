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
from sklearn.pipeline import Pipeline
from mlxtend.plotting import plot_confusion_matrix,plot_decision_regions


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
'''
vectorizer = CountVectorizer(stop_words="english",ngram_range=(1,2), max_features=4000)    
transformer = TfidfTransformer()
'''
## 获取训练集的特征
'''
train_tfidf = transformer.fit_transform(vectorizer.fit_transform(X_train))
train_tfidf = train_tfidf.toarray()
'''
## 获取测试集的特征
'''
test_tfidf = transformer.transform(vectorizer.transform(X_test))
test_tfidf = test_tfidf.toarray()
'''


# 9.2.2 朴素贝叶斯文本分类
## 建立似然为高斯分布的朴素贝叶斯模型
'''
gnb = GaussianNB().fit(train_tfidf, y_train)
gnb_pre = gnb.predict(test_tfidf)
print(classification_report(y_test,gnb_pre))
'''
## 建立似然为多项式分布的朴素贝叶斯模型
'''
mnb = MultinomialNB().fit(train_tfidf, y_train)
mnb_pre = mnb.predict(test_tfidf)
print(classification_report(y_test,mnb_pre))
'''
## 建立似然为伯努利分布的朴素贝叶斯模型
'''
bnb = BernoulliNB().fit(train_tfidf, y_train)
bnb_pre = bnb.predict(test_tfidf)
print(classification_report(y_test,bnb_pre))
'''
## 为方便后面可视化ROC曲线，对标签使用label_binarize进行编码
'''
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
'''

## 使用参数网格搜索，优化模型，寻找更好的参数
'''
## 对建模过程进行封装
bbc_nb = Pipeline([("vect", CountVectorizer(stop_words="english")),
                   ("tfidf", TfidfTransformer()),
                   ("mnb", MultinomialNB()),])
## 定义网格搜索的参数
alpha = [0.001,0.01,0.1,0.5,1,10]
para_grid = {"vect__ngram_range": [(1, 1), (1, 2),(2,3)],
             "vect__max_features":[1000,2000,3000,5000],
             "tfidf__norm": ["l1","l2"],
             "mnb__alpha": alpha}
## 使用3折交叉验证进行搜索
gs_bbc_nb = GridSearchCV(bbc_nb,para_grid,cv=3,n_jobs=4)
gs_bbc_nb.fit(X_train,y_train)
## 得到最好的参数组合
print(gs_bbc_nb.best_params_)
## 使用最好效果的模型对测试集进预测
gs_pre = gs_bbc_nb.best_estimator_.predict(X_test)
## 可视化对测试集的混淆矩阵
lable_names = ["sport","business","politics","tech","entertainment"]
plot_confusion_matrix(confusion_matrix(y_test,gs_pre), figsize=(10,8), class_names=lable_names)
plt.title("朴素贝叶斯份分类（参数搜索）")
plt.show()

print(classification_report(y_test,gs_pre))
'''
## 可视化每个类别的ROC曲线
'''
lable_names = ["sport","business","politics","tech","entertainment"]
colors = ["r","b","g","m","k",]
linestyles =["-", "--", "-.", ":", "-"]
pre_score = gs_bbc_nb.best_estimator_.predict_proba(X_test)
fig  = plt.figure(figsize=(8,7))
for ii, color in zip(range(pre_score.shape[1]), colors):
    ## 计算绘制ROC曲线的取值
    fpr_ii, tpr_ii, _ = roc_curve(y_test_lb[:,ii], pre_score[:,ii])
    plt.plot(fpr_ii, tpr_ii,color = color,linewidth = 2,
             linestyle = linestyles[ii],
             label = "class:"+lable_names[ii])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("假正率")
plt.ylabel("真正率")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.title("每个类别的ROC曲线")
## 添加局部放大图
inset_ax = fig.add_axes([0.3, 0.45, 0.4, 0.4],facecolor="white")
for ii, color in zip(range(pre_score.shape[1]), colors):
    ## 计算绘制ROC曲线的取值
    fpr_ii, tpr_ii, _ = roc_curve(y_test_lb[:,ii], pre_score[:,ii])
    ## 局部放大图
    inset_ax.plot(fpr_ii, tpr_ii,color = color,linewidth = 2,
                  linestyle = linestyles[ii])
    inset_ax.set_xlim([-0.01,0.1])
    inset_ax.set_ylim([0.88,1.01])
    inset_ax.grid()
plt.show()
'''

