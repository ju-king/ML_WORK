# ML_WORK
## Python题目分类：选择、判断、编程、填空
### 环境配置
#### python>=3.7
#### ----------------- -------
#### lightgbm          3.3.2
#### numpy             1.21.6
#### openpyxl          3.0.10
#### pandas            1.3.5
#### pkuseg            0.0.25
#### scikit-learn      1.0.2
#### tqdm              4.64.0
#### gensim            4.2.0
## modeling.py实现题型分类
## recommendation.py实现题目推荐
## show_feature.py可视化分析文本特征
## 大致思路
### 首先读带有标签的数据集（共3519条），经过pkuseg分词后过一次TFIDF（PCA降维为128）和一次W2V（dim=128，句向量使用平均词向量表示），拼接TFIDF和W2V目的是使用TFIDF补充W2V，然后将标签通过encoder，最后通过分类器LightGBM进行多分类任务。
### 推荐使用w2v实现余弦相似度计算，根据余弦相似度排序推荐排名前3的题目。
## 分类AUC...  0.9909793103012347
