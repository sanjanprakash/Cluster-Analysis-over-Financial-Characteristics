import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import *
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA
import seaborn
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import copy
from sklearn.externals.joblib import parallel_backend
import sys
from scipy.spatial.distance import cdist

plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname('./'))


# Reading 
picklePath = './pickle/tenYears.pkl'
df = pd.read_pickle(picklePath)
print ('Reading done!')

#Dropping NAN
print ("Input df dimention " + str(df.shape))
df = df.dropna()
print ('Droped NAN')

# Creating Target Variable column based on year and ticker symbol
df['Target Variable'] = (df['Data Year - Fiscal'].astype(int)).astype(str) + " " + df['Ticker Symbol']


# Dropping unecessary and redundant columns
df.drop(['Data Year - Fiscal'], axis=1,inplace=True)
df.drop(['Ticker Symbol', 'CUSIP', 'Company Name', 'State/Province','Data Date', 'Global Company Key'], axis=1,inplace=True)


# Min max scaler from sklearn to normalise the columns
min_max_scaler = preprocessing.MinMaxScaler()
# Selecting all the feature variable columns
dfValueCols = df.loc[:, df.columns != 'Target Variable']
np_scaled = min_max_scaler.fit_transform(dfValueCols)
df_normalized = pd.DataFrame(np_scaled,columns=dfValueCols.columns,
								index = dfValueCols.index)

# selecting the target variable column
symbolAndYear = df['Target Variable'].to_frame()

# Final normalised dataset
df = symbolAndYear.merge(df_normalized, how='outer', left_index=True, right_index=True)
# Selecting all the feature variable columns
dfValueCols = df.loc[:, df.columns != 'Target Variable']

# Applying PCA based dimentionality reduction
pca = PCA(n_components=2)
pca.fit(dfValueCols)
columns = ['pca_%i' % i for i in range(2)]
pca_df = pd.DataFrame(pca.transform(dfValueCols), columns=columns, index=df.index)
final = symbolAndYear.merge(pca_df, how='outer', left_index=True, right_index=True)

# Visualise data after using PCA to bring it to 2D
fig, ax = plt.subplots()
final.plot(kind='scatter',x='pca_0',y='pca_1',color='red',ax=ax)
plt.title("Input data plot after dimentionality reduction")
plt.show()

plt.clf()

# k means determine k using elbow method
distortions = []
# maximum iteration for K means elbow methods
maxIter = 10
K = range(1,maxIter+1)
for k in K:
	print("Analysing with "+str(k)+" centroids")
	kmeanModel = KMeans(n_clusters=k).fit(dfValueCols)
	kmeanModel.fit(dfValueCols)
	distortions.append(sum(np.min(cdist(dfValueCols, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dfValueCols.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# KMEANS applied on initial dimentionality data
# We decided to take 20 centroids after the using the elbow method for upto 100 centorids
kmeans = KMeans(n_clusters=20).fit(dfValueCols) 
df['KMeans_Labels'] = kmeans.labels_
df_Kmeans = copy.deepcopy(df)
pca_df['KMeans_Labels'] = kmeans.labels_
plt.scatter(pca_df['pca_0'], pca_df['pca_1'], c= kmeans.labels_.astype(float), s=50, alpha=0.5,cmap="viridis")
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('K Means with 20 centroids')
plt.show()


# Using DBSCAN for learning purposes
# Setting hyperparemeters for DBSCAN
epsValues = [0.05,0.1,0.2,0.05,0.1,0.2,0.05,0.1,0.2]
min_samples_Value = [1,1,1,5,5,5,9,9,9]
# Other possible values
# epsValues = [0.125,0.25,0.5,0.1,0.123,0.2,0.3,0.4,0.5]
# min_samples_Value = [1,2,3,4,5,6,7,8,9]
# epsValues = [1,2,3,1,2,3,1,2,3]
# min_samples_Value = [2,2,2,3,3,3,5,5,5]

# Loop to evaluate DBSCAN on various different hyperparameters and plot it
i = 1
for esp, min_samples in zip(epsValues,min_samples_Value):
	dbscan = DBSCAN(eps=esp, min_samples = min_samples)
	dbscanClusters = dbscan.fit_predict(dfValueCols)
	plt.subplot(3,3,i)
	i += 1
	plt.scatter(pca_df['pca_0'], pca_df['pca_1'], c= dbscanClusters,cmap="viridis")
	num = len(np.unique(dbscanClusters))
	plt.title("Distance: "+str(esp)+"   "+"min_samples: "+str(min_samples)+ "   "+"Clusters: "+str(num))

plt.show()

# Get DBSCAN prediction based on most appropriate cluster infered from the graph
dbscan = DBSCAN(eps=0.05, min_samples = 5)
dbscanClusters = dbscan.fit_predict(dfValueCols)
df['DBSCAN_Labels'] = dbscanClusters
pca_df['DBSCAN_Labels'] = dbscanClusters
plt.scatter(pca_df['pca_0'], pca_df['pca_1'], c= dbscanClusters,cmap="viridis")
plt.title("DBSCAN based clusters eps=0.05, min_samples = 5")
plt.show()


# Wrapper methods to get the most important features from the clusters generated from KMEANS
model = LogisticRegression()
rfe = RFE(model, 30)
# nparrayInputs = (df_Kmeans.loc[:, df.columns != ['Target Variable', 'KMeans_Labels']]).values
nparrayDF = df[df.columns[~df.columns.isin(['Target Variable', 'KMeans_Labels'])]]
nparrayInputs = df[df.columns[~df.columns.isin(['Target Variable', 'KMeans_Labels'])]].values
nparrayTarget= df_Kmeans['KMeans_Labels'].values
fit = rfe.fit(nparrayInputs, nparrayTarget)
print ("All features:")
print (df_Kmeans.columns)
print ("Top 30 features")
print (nparrayDF.loc[:, fit.support_]).columns
sys.exit(1)