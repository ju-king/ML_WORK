import pkuseg
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


def read_csv():
    data = pd.read_excel('data/data_all_label.xlsx')
    return data


def cut_word(data):
    seg = pkuseg.pkuseg()
    sentence = []
    for i in tqdm(data.values):
        text = seg.cut(i)
        sentence.append(' '.join(text))
    return sentence


def label_encoder(data):
    label = LabelEncoder()
    integer_encoded = label.fit_transform(data.values)
    return integer_encoded


def get_feature(data):
    vectorizer = CountVectorizer()
    # 计算每一个词语的TF-IDF权值
    tf_idf_transformer = TfidfTransformer()
    # 计算每一个词语出现的次数#将文本转换为词频并计算tf-idf;fit_transform()方法用于计算每一个词语出现的次数
    X = vectorizer.fit_transform(data)
    tf_idf = tf_idf_transformer.fit_transform(X)

    ##加载PCA模型并训练、降维
    pca = PCA(n_components=128)
    tf_idf_pca = pca.fit_transform(tf_idf.toarray())
    return tf_idf_pca


def train(data):
    param = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'auc_mu',
        'num_leaves': 300,
        'min_data_in_leaf': 500,
        'learning_rate': 0.01,
        # 'feature_fraction': 0.8,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'lambda_l1': 0.4,
        'lambda_l2': 0.5,
        # 'min_gain_to_split': 0.2,
        'verbose': -1,
        'num_threads': 4,
    }

    # 五折交叉验证
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    oof = np.zeros([len(data), 4])
    feats = [f for f in data.columns if f not in ['label']]
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data[feats], data['label'])):
        print("fold n°{}".format(fold_ + 1))
        train = lgb.Dataset(data.loc[trn_idx, feats],
                            data.loc[trn_idx, 'label'])
        val = lgb.Dataset(data.loc[val_idx, feats],
                          data.loc[val_idx, 'label'])

        clf = lgb.train(param,
                        train,
                        valid_sets=[train, val],
                        num_boost_round=5000,
                        verbose_eval=100,
                        early_stopping_rounds=100)
        oof[val_idx] = clf.predict(data.loc[val_idx, feats], num_iteration=clf.best_iteration)
    print('AUC... ', roc_auc_score(data['label'], oof, multi_class='ovr'))


if __name__ == '__main__':
    data = read_csv()
    label = label_encoder(data['标签'])
    tf_idf_pca = get_feature(cut_word(data['题目']))

    clos = []
    for i in range(128):
        clos.append('_pca_' + str(i))
    print(clos)

    df_data = pd.DataFrame(tf_idf_pca, columns=clos)
    df_data['label'] = label
    df_data.to_csv('output/feature_pca128.csv', index=None)
    train(df_data)
