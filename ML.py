import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import SkLearnHelper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from DataInit import splitData
from sklearn.decomposition import PCA
from Preprocess import preprocess

df1 = pd.read_csv('train.csv')
# df2 = pd.read_csv('test.csv')
# df = pd.concat([df1, df2], axis=0)

X = df1

col_names = X.columns
y = np.array(X.pop('label'))

model = SelectFromModel(RandomForestClassifier().fit(X, y), prefit=True)
X_new = pd.DataFrame(model.transform(X))


#pca = PCA(n_components=0.95, svd_solver='full').fit(X_new)
#print(len(pca.explained_variance_ratio_))

preprocess(X_new, y, standardization=True, normalization=False)

# SkLearnHelper.fit_classifier_gridSearch(SkLearnHelper.zipped_clf, X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# SkLearnHelper.fit_classifier_default(SkLearnHelper.zipped_clf,X_train,y_train, X_test, y_test)


# Parameters of pipelines can be set using ‘__’ separated parameter names:

# search = GridSearchCV(pipe, param_grid, n_jobs=-1)
# search.fit(X_digits, y_digits)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)
