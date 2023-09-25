#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import *
import scipy.cluster.hierarchy as spc

# load emotionnet, show corr matrrix
emotionnet_aus = pickle.load(open('../datasets/emotionNet/aus_openface.pkl', 'rb'))
df_aus = pd.DataFrame(list(emotionnet_aus.values()))

df_aus_corr = df_aus.corr()
plt.matshow(df_aus_corr)
plt.show()


#%%
# load cancer data
from sklearn import preprocessing
df_cancer = pd.read_csv("../_datasets/wisconsin_breast_cancer.csv", encoding='utf-8')
scaler = preprocessing.MinMaxScaler(feature_range=(0,3)).fit(df_cancer.iloc[:, 2:19])
df_cancer_scaled  = scaler.transform(df_cancer.iloc[:, 2:19])

#%% reorder features - based on the IGTD minimalizatin
from scritps.IGTD_Functions import generate_feature_distance_ranking
from scritps.IGTD_Functions import IGTD_absolute_error

fdr_target = generate_feature_distance_ranking(df_aus.values)[0]
fdr_source = generate_feature_distance_ranking(df_cancer_scaled)[0]
igtd_ranking, err, _ = IGTD_absolute_error(fdr_source, fdr_target)
min_ranking_index = np.argmin(err)
#%%
df_cancer_aus = pd.DataFrame(df_cancer_scaled)[igtd_ranking[min_ranking_index, :]]
id_col = df_cancer['diagnosis'] + "_" + df_cancer['id'].astype(str)
df_cancer_aus.insert(0, 'id', id_col)
df_cancer_aus.to_csv('../datasets/generate_test/cancer_aus.csv', header=False, index=False)

#%%
from skfeature.function.similarity_based import SPEC

scaler = preprocessing.MinMaxScaler().fit(df_cancer.iloc[:, 2:])
df_cancer_scaled_all  = scaler.transform(df_cancer.iloc[:, 2:])
cancer_spec_ord = SPEC.spec(df_cancer_scaled_all)
#%%
scaler = preprocessing.MinMaxScaler().fit(df_aus)
df_aus_scaled  = scaler.transform(df_aus)
aus_spec_ord = SPEC.spec(df_aus_scaled[0:5000])