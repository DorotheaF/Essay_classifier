import pandas as pd
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
import json

def fit_model(X_train, y_train, X_test, y_test):

    # classifier = XGBClassifier(objective='multi:softmax', max_depth=4, learning_rate=.8, n_estimators=100)
    # classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    # classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.8, max_depth=1, random_state=0)
    classifier = RandomForestClassifier(max_depth=5, random_state=0)
    # classifier = sklearn.linear_model.LogisticRegression(multi_class='multinomial', random_state=0, max_iter=3000)
    # classifier = sklearn.naive_bayes.GaussianNB()

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    return matrix

def run_models(num_folds, experiment = "ITA_ITA"):
    data = get_data()
    dataframe = data[0].copy()
    dataframe['level'] = dataframe['level'].replace('low', 0).replace('medium', 1).replace('high', 2)
    # print(dataframe.columns.to_list())
    X = dataframe[[col for col in dataframe.columns.to_list() if col != 'level']]
    y = dataframe['level']

    folds = StratifiedKFold(num_folds, shuffle=True, random_state=0)

    model_predictions = {}

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        predictions = fit_model(X.iloc[train_index], y.iloc[train_index], X.iloc[test_index], y.iloc[test_index])
        model_predictions[i] = predictions

    total_matrix = model_predictions[0]
    
    
    for i in range(1, num_folds):
        total_matrix += model_predictions[i]

    c1 = total_matrix[0][0]
    c2, c3 = 0, 0

    if len(total_matrix) > 1 and len(total_matrix[1]) > 1:
        c2 =  total_matrix[1][1]
    
    if len(total_matrix) > 2 and len(total_matrix[2]) > 2:
        c3 =  total_matrix[2][2]

    accuracy = (c1 + c2 + c3) / len(X)
    # print(features)
    print(accuracy)
    print(total_matrix)

    res = {"accuracy": accuracy, "cf": total_matrix.tolist()}
    with open(f"./results/{experiment}.json", "w") as f:
        json.dump(res, f)

def run_models_different_lang():
    data = get_data()
    dataframe, val = data[0].copy(), data[1].copy()
    dataframe['level'] = dataframe['level'].replace('low', 0).replace('medium', 1).replace('high', 2)
    val['level'] = val['level'].replace('low', 0).replace('medium', 1).replace('high', 2)
    X = dataframe[[col for col in dataframe.columns.to_list() if col != 'level']]
    y = dataframe['level']
    X_test = val[[col for col in val.columns.to_list() if col != 'level']]
    y_test = val['level']

    total_matrix = fit_model(X, y, X_test, y_test)

    c1 = total_matrix[0][0]
    c2, c3 = 0, 0

    if len(total_matrix) > 1 and len(total_matrix[1]) > 1:
        c2 =  total_matrix[1][1]
    
    if len(total_matrix) > 2 and len(total_matrix[2]) > 2:
        c3 =  total_matrix[2][2]

    accuracy = (c1 + c2 + c3) / len(X_test)
    print(accuracy)
    print(total_matrix)

    res = {"accuracy": accuracy, "cf": total_matrix.tolist()}
    with open(f"./results/ITA_OTHERS.json", "w") as f:
        json.dump(res, f)



def get_lang_files(lang = 'ITA'):
    df = pd.read_excel("./data/TOEFL prompt 4 - dev.xlsx", sheet_name = 'Sheet3')
    ita_ids = []
    for l, i in zip(df['LANG'].tolist(), df['ID'].tolist()):
        if l == lang:
            ita_ids.append(i)
    return [int(id.replace('.txt','')) for id in ita_ids]


def get_difficulty_ids(diff = 'low'):
    df = pd.read_excel("./data/TOEFL prompt 4 - dev.xlsx", sheet_name = 'Sheet3')
    diff_ids = []
    for d, i in zip(df['DIFFICULTY'].tolist(), df['ID'].tolist()):
        if d == diff:
            diff_ids.append(i)

    return [int(id.replace('.txt','')) for id in diff_ids]


def get_data():
    drop_file_ids = [95152, 250580, 1020096]
    diff_ids = get_lang_files('ITA')
    df = pd.read_excel("./data/TOEFL Annotation_mine.xlsx")
    df = df.drop(columns=['text_standard', 'essay'])
    df = df[~df['name'].isin(drop_file_ids)]
    # df = df[df['name'].isin(diff_ids)]
    # # print(df[features])
    return df[df['name'].isin(diff_ids)], df[~df['name'].isin(diff_ids)]

num_folds = 5

run_models_different_lang()
run_models(num_folds)
