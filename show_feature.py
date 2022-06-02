
# 导入chain方法用于扁平化列表
import jieba
from itertools import chain

# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt




data = pd.read_excel('data/data_all_label.xlsx')
# 在训练数据中添加新的句子长度列, 每个元素的值都是对应的句子列的长度
data["sentence_length"] = list(map(lambda x: len(x), data['题目']))

# 绘制句子长度列的数量分布图
sns.countplot("sentence_length", data=data)
# 主要关注count长度分布的纵坐标, 不需要绘制横坐标, 横坐标范围通过dist图进行查看
plt.xticks([])
plt.savefig('output/sentence_length.jpg')
plt.show()


# 绘制dist长度分布图
sns.distplot(data["sentence_length"])

# 主要关注dist长度分布横坐标, 不需要绘制纵坐标
plt.yticks([])
plt.savefig('output/dist.jpg')
plt.show()




# 进行训练集的句子进行分词, 并统计出不同词汇的总数
train_vocab = set(chain(*map(lambda x: jieba.lcut(x), data['题目'])))
print("数据集共包含不同词汇总数为：", len(train_vocab))

# 使用jieba中的词性标注功能
import jieba.posseg as pseg


def get_a_list(text):
    """用于获取形容词列表"""
    # 使用jieba的词性标注方法切分文本,获得具有词性属性flag和词汇属性word的对象,
    # 从而判断flag是否为形容词,来返回对应的词汇
    r = []
    for g in pseg.lcut(text):
        if g.flag == "a":
            r.append(g.word)
    return r


# 导入绘制词云的工具包
from wordcloud import WordCloud


def get_word_cloud(keywords_list):
    # 实例化绘制词云的类, 其中参数font_path是字体路径, 为了能够显示中文,
    # max_words指词云图像最多显示多少个词, background_color为背景颜色
    wordcloud = WordCloud(font_path="C:/Windows/Fonts/SimHei.ttf", max_words=100, background_color="white")
    # 将传入的列表转化成词云生成器需要的字符串形式
    keywords_string = " ".join(keywords_list)
    # 生成词云
    wordcloud.generate(keywords_string)

    # 绘制图像并显示
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('output/wordcloud.jpg')
    plt.show()





# 每个句子的形容词
train_p_a_vocab = chain(*map(lambda x: get_a_list(x), data['题目']))


# 调用绘制词云函数
get_word_cloud(train_p_a_vocab)

