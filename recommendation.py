import modeling
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


data = pd.read_excel('data/data_all_label.xlsx')
data_vec = pd.read_csv('output/feature_pca128+embed128.csv')
model = Word2Vec.load('output/word2vec2.model')

inputs = input()

sentences = modeling.cut_word([inputs])

M=[]
for words in sentences:
    for word in  words:
        M.append(model.wv[word])
M = np.array(M)
v = M.sum(axis=0)
sentence_vectors=(v/len(sentences))

con_sim=[]
for i in range(len(data_vec)):
    x1 = np.array(data_vec.loc[i][129:]).reshape(1, 128)
    x2 = np.array(sentence_vectors).reshape(1,128)
    con_sim.append(cosine_similarity(x1, x2))
data['con_sim'] = con_sim

recom_data=data.sort_values(by='con_sim', ascending=False)[:3]

for i in recom_data.index:
    print(recom_data.loc[i][:7])




