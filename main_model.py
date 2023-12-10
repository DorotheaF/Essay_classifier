import numpy as np
import pandas as pd
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

def fit_model(X_train, y_train, X_test, y_test):

    # classifier = XGBClassifier(objective='multi:softmax', max_depth=4, learning_rate=.8, n_estimators=100)
    # classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    # classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.8, max_depth=1, random_state=0)
    # classifier = RandomForestClassifier(max_depth=5, random_state=0)
    # classifier = sklearn.linear_model.LogisticRegression(multi_class='multinomial', random_state=0, max_iter=3000)
    classifier = sklearn.naive_bayes.GaussianNB()

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    return matrix

def run_models(features, num_folds):
    dataframe = pd.read_excel('data/TOEFL Annotation_1.xlsx')
    dataframe['level'] = dataframe['level'].replace('low', 0).replace('medium', 1).replace('high', 2)
    X = dataframe[features]
    y = dataframe['level']
    print(len(X), len(y))
    folds = StratifiedKFold(num_folds, shuffle=True, random_state=0)

    model_predictions = {}

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        predictions = fit_model(X.loc[train_index], y.loc[train_index], X.loc[test_index], y.loc[test_index])
        model_predictions[i] = predictions

    total_matrix = model_predictions[0]

    for i in range(1, num_folds):
        total_matrix += model_predictions[i]
    print(model_predictions)
    print(total_matrix)
    accuracy = (total_matrix[0][0] + total_matrix[1][1] + total_matrix[2][2]) / 79
    print(features)
    print(accuracy)
    print(total_matrix)



# Full features list: ['count', 'cont', 'retain', 'shift', 'one_mention',
#        'multi_mention', 'flesch_reading_ease', 'difficult words', 'dale_chall_readability_score',
#        'textstat difficult words', 'arc_length', 'average_height', 'height',
#        'num_edus']

features = ['count', 'cont', 'retain', 'shift', 'one_mention', 'multi_mention', 'flesch_reading_ease']
num_folds = 5

run_models(features, num_folds)
