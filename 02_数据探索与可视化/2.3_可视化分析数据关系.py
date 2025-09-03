import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic
import plotly.express as px
from pandas.plotting import parallel_coordinates
from wordcloud import WordCloud
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from vega_datasets import data
import numpy as np

# 处理matplotlib中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  
sns.set_theme(font='Kaiti', style='ticks', font_scale=1.4)

# 2.3.1 连续变量间关系可视化
Iris = pd.read_csv("02_数据探索与可视化/data/Iris.csv")
Iris2 = Iris.drop(['Id', "Species"], axis=1)
## （1）两个连续变量之间的可视化
'''散点图
plt.figure(figsize=(10, 6))
plt.title('散点图')
plt.scatter(Iris2['SepalLengthCm'], Iris2['SepalWidthCm'], c='blue', marker='o')
sns.scatterplot(data=Iris2, x='SepalLengthCm', y='SepalWidthCm', s=50)
plt.grid()
plt.show()
'''
'''2D密度曲线
sns.jointplot(data=Iris2, x='SepalLengthCm', y='SepalWidthCm', kind='kde', fill=True, cmap='Blues', height=7)
plt.grid()
plt.show()
'''
'''直方图
Iris2.iloc[:, 0:2].plot(kind='hist', bins=30, figsize=(10, 6), edgecolor='black', alpha=0.7)
plt.title('直方图')
plt.show()
'''

## （2）多个连续变量之间的可视化
'''气泡图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=Iris2, x='SepalLengthCm', y='SepalWidthCm', size='PetalLengthCm', sizes=(20, 200), palette='muted')
plt.title('气泡图')
plt.legend(loc="center right", bbox_to_anchor=(1.13, 0.5))
plt.grid()
plt.show()
'''
'''小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(data=Iris2.iloc[:, 0:4], palette='Set3', bw=0.5)
plt.title('小提琴图')
plt.grid()
plt.show()
'''
'''蒸汽图
# 将鸢尾花宽数据转化成长数据
Irislong = Iris.melt(["Id", "Species"], var_name="Measurement_type", value_name="Value")
#使用蒸汽图可视化
chart = alt.Chart(Irislong).mark_area().encode(
    alt.X('Id:Q'),
    alt.Y('Value:Q', stack='center', axis=None),
    alt.Color('Measurement_type:N'),
).properties(width=500, height=300, title='蒸汽图')
chart.save('02_数据探索与可视化/data/2.3_可视化分析数据关系蒸汽图.html')
'''

# 2.3.2 分类变量间关系可视化
Titanic = pd.read_csv("02_数据探索与可视化/data/Titanic数据.csv")
## （1）两个分类变量之间的可视化
'''卡方检验
tab = pd.crosstab(Titanic['Embarked'], Titanic['Survived'])
c,p,_,_ = chi2_contingency(tab.values)
'''
'''马赛克图
mosaic(Titanic, ['Embarked', 'Survived'], title='马赛克图')
plt.show()
'''
## （2）多个分类变量之间的可视化
'''树图
Titanic["Titanic"] = "Titanic"
Titanic["Value"] = 1
print(Titanic.head())
fig = px.treemap(Titanic, path=['Titanic', 'Survived', 'Sex', 'Embarked'], 
                 values='Value', color = 'Fare',
                 color_continuous_scale='RdBu',
                 width=800, height=500,)
fig.write_html("02_数据探索与可视化/data/2.3_可视化分析数据关系树图.html")
'''

# 2.3.3 连续变量与分类变量间关系可视化
Irislong = Iris.melt(["Id", "Species"], var_name="Measurement_type", value_name="Value")
## （1）一个连续变量与一个分类变量
'''分组箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=Irislong, x='Species', y='Value', palette='Set3')
plt.title('分组箱线图')
plt.show()
'''
'''分面密度图
# 区分品种显示
chart = alt.Chart(Irislong).transform_density(
            density='Value', bandwidth=0.3, 
            groupby=['Measurement_type', 'Species'], extent=[0, 8]
        ).mark_area().encode(
            alt.X('value:Q'),alt.Y('density:Q'),
            alt.Color('Species:N'),
            alt.Row('Measurement_type:N')
        ).properties(width=500, height=80, title='分面密度图')
chart.save('02_数据探索与可视化/data/2.3_可视化分析数据关系分面密度图_1.html')

# 不区分品种显示
chart = alt.Chart(Irislong).transform_density(
            density='Value', bandwidth=0.3, 
            groupby=['Measurement_type'], extent=[0, 8]
        ).mark_area().encode(
            alt.X('value:Q'),alt.Y('density:Q'),
            alt.Row('Measurement_type:N')
        ).properties(width=500, height=80, title='分面密度图')
chart.save('02_数据探索与可视化/data/2.3_可视化分析数据关系分面密度图_2.html')
'''
## （2）两个分类变量的一个连续变量
'''分组箱线图
plt.figure(figsize=(10,6))
sns.boxplot(data=Irislong, x='Measurement_type', y='Value', hue='Species')
plt.legend(loc=1)
plt.title('分组箱线图')
plt.show()
'''
## （3）两个分类变量和两个连续变量
'''分面散点图
g = sns.FacetGrid(data = Titanic, row="Survived", col="Sex", margin_titles=True, height=3, aspect=1.4)
g.map(sns.scatterplot,"Age" ,"Fare",)
plt.show()
'''
## （4）一个分类变量和多个连续变量
'''平行坐标图
plt.figure(figsize=(10,6))
parallel_coordinates(Iris.iloc[:,1:6], "Species",alpha = 0.8)
plt.title("平行坐标图")
plt.show()
'''
'''矩阵散点图
sns.pairplot(Iris.iloc[:,1:6], hue="Species", height=2, aspect=1.2, diag_kind="kde", markers=["o", "s", "D"])
plt.show()
'''
'''分组气泡图
sns.relplot(data = Iris, x="SepalWidthCm", y="PetalWidthCm", 
            hue="Species", size = "SepalLengthCm", sizes = (20,200),
            palette="muted", height=6, aspect = 1.4)
plt.title("分组气泡图")
plt.show()
'''

# 2.3.4 其它数据类型可视化分析
'''时间序列数据
opsd = pd.read_csv("02_数据探索与可视化/data/OpenPowerSystemData.csv")
opsd.head()
opsd.plot(kind = "line",x = "Date",y = "Solar",figsize = (10,6)) # 折线图
plt.ylabel("Value")
plt.title("时间序列曲线")
plt.show()
'''
'''文本数据
## 准备数据
TKing = pd.read_csv("02_数据探索与可视化/data/三国演义分词后.csv")
## 计算每个词语出现的频次
TK_fre = TKing.x.value_counts()
TK_fre = pd.DataFrame({"word":TK_fre.index,
                       "Freq":TK_fre.values})
## 去除出现次数较少的词语
TK_fre = TK_fre[TK_fre.Freq > 100]
TK_fre
## 可视化词云
## 将词和词频组成字典数据准备
worddict = {}
for key,value in zip(TK_fre.word,TK_fre.Freq):
     worddict[key] = value  
## 生成词云
redcold = WordCloud(font_path=".venv_ml/Library/Fonts/Microsoft/SimHei.ttf",
                     margin=5,width=1800, height=1000,
                     max_words=400, min_font_size=5, 
                     background_color='white',
                     max_font_size=250,)
redcold.generate_from_frequencies(frequencies=worddict)
plt.figure(figsize=(10,7))
plt.imshow(redcold)
plt.axis("off")
plt.show()
'''
'''社交网络数据
## 读取网络数据
karate = pd.read_csv("02_数据探索与可视化/data/karate.csv")
## 网络图数据可视化
plt.figure(figsize=(12,8))
## 生成社交网络图
G=nx.Graph()
## 为图像添加边
for ii in karate.index:
    G.add_edge(karate.From[ii], karate.to[ii], weight = karate.weight[ii])
## 根据权重大小定义2种边
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] > 3.5]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] < 3.5]
## 图的布局方式
# pos=graphviz_layout(G, prog="fdp")
pos=nx.circular_layout(G)

# 可视化图的节点
nx.draw_networkx_nodes(G,pos,alpha=0.4,node_size=20)

# 可视化图的边
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                       width=2,alpha=0.5,edge_color= "red")
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                       width=2,alpha=0.5,edge_color="blue",style='dashed')

# 为节点添加标签
nx.draw_networkx_labels(G,pos,font_size = 14)
plt.axis('off')
plt.title("空手道俱乐部人物关系")
plt.show() 
'''
'''地图数据
states = alt.topo_feature(data.us_10m.url, feature='states')
## 机场位置和数量数据
airports = pd.read_csv("02_数据探索与可视化/data/airports.csv")
airports = airports.groupby("state").agg(latitude = ("latitude","mean"),
                                         longitude = ("longitude","mean"),
                                         number = ("state",np.size))
airports["state2"] = airports.index.values
## 气泡地图
## 美国地图背景
background = alt.Chart(states).mark_geoshape(fill="lightblue",stroke="white"
            ).properties(width=500,height=300).project("albersUsa")
## 机场的位置和气泡
points = alt.Chart(airports).mark_circle().encode(
                    longitude="longitude:Q",
                    latitude="latitude:Q",
                    size=alt.Size("number:Q", title="机场数量"),
                    color=alt.value("red"),
                    tooltip=["state2:N","number:Q"]
                  ).properties(title="美国机场数量")
## 可视化背景和点
(background + points).save('02_数据探索与可视化/data/2.3_可视化分析数据机场气泡图.html')
'''