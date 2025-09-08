from sklearn.datasets import load_wine
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif

# 导入酒的多分类数据集
wine_x, wine_y = load_wine(return_X_y=True)
#print(wine_x.shape)
#print(np.unique(wine_y, return_counts=True))

# 3.3.1 基于统计方法
'''剔除低方差的特征
VTH = VarianceThreshold(threshold=0.5)
VTH_wine_x = VTH.fit_transform(wine_x)
print("剔除低方差特征后数据集的形状：", VTH_wine_x.shape)
print(VTH.variances_ > 0.5)
'''
'''通过方差分析的F统计量选择K个特征
KbestF = SelectKBest(score_func=f_classif, k=5)
KbestF_wine_x = KbestF.fit_transform(wine_x, wine_y)
print("通过方差分析选择K个特征后数据集的形状：", KbestF_wine_x.shape)
'''
'''通过卡方值选择K个变量
KbestChi2 = SelectKBest(score_func=chi2, k=5)
KbestChi2_wine_x = KbestChi2.fit_transform(wine_x, wine_y)
print("通过卡方值选择K个特征后数据集的形状：", KbestChi2_wine_x.shape)
'''
'''通过互信息法选择K个变量
KbestMI = SelectKBest(score_func=mutual_info_classif, k=5)
KbestMI_wine_x = KbestMI.fit_transform(wine_x, wine_y)  
print("通过互信息法选择K个特征后数据集的形状：", KbestMI_wine_x.shape)
'''

# 3.3.2 基于递归消除特征法
'''使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
from sklearn.feature_selection import RFE,RFECV
from sklearn.ensemble import RandomForestClassifier
#设置基模型为随机森林
model = RandomForestClassifier(random_state=0) #设置基模型为随机森林
rfe = RFE(estimator = model,n_features_to_select = 9) #选择9个最佳特征变量
rfe_wine_x = rfe.fit_transform(wine_x, wine_y) #进行RFE递归
print("特征是否被选中:\n",rfe.support_)  
print("获取的数据特征尺寸:",rfe_wine_x.shape)
## 借助5折交叉验证最少选择5个最佳特征变量
rfecv = RFECV(estimator = model,min_features_to_select = 5, cv = 5) 
rfecv_wine_x = rfecv.fit_transform(wine_x, wine_y) #进行RFE递归
print("特征是否被选中:\n",rfecv.support_)  
print("获取的数据特征尺寸:",rfecv_wine_x.shape)
'''

# 3.3.3 基于机器学习的方法
## 根据特征的重要性权重选择特征
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

'''利用随机森林模型进行特征的选择
rfc = RandomForestClassifier(n_estimators=100,random_state=0)
rfc = rfc.fit(wine_x,wine_y) # 使用模型拟合数据
## 定义从模型中进行特征选择的选择器
sfm = SelectFromModel(estimator=rfc, ## 进行特征选择的模型
                      prefit = True, ## 对模型进行预训练
                      max_features = 10,##选择的最大特征数量
                     )
## 将模型选择器作用于数据特征
sfm_wine_x = sfm.transform(wine_x)
print(sfm_wine_x.shape)
'''
'''在特征的选择时还可以利用L1范数进行选择
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

## 在构建支持向量机分类时使用L1范数约束
svc = LinearSVC(penalty="l1",dual=False,C = 0.05)
svc = svc.fit(wine_x,wine_y)
## 定义从模型中进行特征选择的选择器
sfm = SelectFromModel(estimator=svc, ## 进行特征选择的模型
                      prefit = True, ## 对模型进行预训练
                      max_features = 10,##选择的最大特征数量
                     )
## 将模型选择器作用于数据特征
sfm_wine_x = sfm.transform(wine_x)
print(sfm_wine_x.shape)
'''