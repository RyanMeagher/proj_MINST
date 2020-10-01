from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from paramGrid import createParamGrid
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

classifier_names = ["RandForest", "KNN", "SVM", "NB", "XgBoost", "Logistic Regression"]
classifiers = [
    RandomForestClassifier(n_jobs=-1),
    KNeighborsClassifier(n_jobs=-1),
    SVC(),
    GaussianNB(),
    GradientBoostingClassifier(),
    LogisticRegression(max_iter=100, n_jobs=-1)]

zipped_clf = zip(classifier_names, classifiers)


def fit_classifier_default(classGrid, x_train, y_train, x_test, y_test):
    result = []
    for n, c in classGrid:
        pipe = Pipeline([
            ('standardize', StandardScaler()),
            #('feature_selection', SelectFromModel(RandomForestClassifier())),
            'feature reduction', PCA(n_components=5),
            (f'{n} classifier', c)
        ])
        print(f"Validation result for {n}")
        model_fit = pipe.fit(x_train, y_train)
        y_pred = model_fit.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy score: {0:.2f}%".format(accuracy * 100))
        result.append((n, accuracy))
    return result


def fit_classifier_gridSearch(classGrid, X, y, num_cross_val=2):
    modelDict = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    paramGrid = createParamGrid(X)

    for n, c in classGrid:
        print(f'Initializaing grid search to find best parameters of {n} classifier')

        pipe = Pipeline([
            ('standardize', StandardScaler()),
            (f'{n} classifier', c)
        ])

        search = GridSearchCV(pipe, paramGrid[f'{n} classifier'], cv=num_cross_val, n_jobs=-1)
        print(f'The best parameters for {n} classifier are are being searched for via GridSearchCV')

        model_fit = search.fit(X_train, y_train)
        print(search)
        y_pred = model_fit.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("accuracy score: {0:.2f}%".format(accuracy * 100))

        # add the best parameters found via GridSearchCV and also the
        modelDict[n] = [search, y_pred, accuracy]

    return accuracy
