import numpy as np
import pandas as pd
import sklearn.naive_bayes
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def run_NB_model(X_train, y_train, X_test, y_test):

    classifier = sklearn.naive_bayes.GaussianNB()
    classifier.fit(X, y)

    y_pred = classifier.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

    # print("For the features used: " + str(features))
    # print("The accuracy of the classifier is:", accuracy)
    # print(matrix)
    return matrix

def run_RF_model(features, dataframe):

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test)

    accuracy = np.mean(y_pred == Y_test)
    matrix = sklearn.metrics.confusion_matrix(Y_test, y_pred)

    print("For the features used: " + str(features))
    print("The accuracy of the classifier is:", accuracy)
    print(matrix)

dataframe = pd.read_excel('data/TOEFL Annotation_1.xlsx')
# print(dataframe.columns)

dataframe['level'] = dataframe['level'].replace('low', '0').replace('medium', '1').replace('high', '2')
# Full features list: ['count', 'cont', 'retain', 'shift', 'one_mention',
#        'multi_mention', 'flesch_reading_ease', 'difficult words', 'dale_chall_readability_score',
#        'textstat difficult words', 'arc_length', 'average_height', 'height',
#        'num_edus']

features = ['count', 'cont', 'retain', 'shift', 'one_mention', 'multi_mention', 'difficult words']
num_folds = 5

dataframe = dataframe.sample(frac = 1, random_state=3)
X = dataframe[features]
y = dataframe['level']
folds = StratifiedKFold(num_folds, shuffle=True)

model_predictions = {}

for i, (train_index, test_index) in enumerate(folds.split(X, y)):
    predictions = run_NB_model(X.loc[train_index], y.loc[train_index], X.loc[test_index], y.loc[test_index])
    model_predictions[i] = predictions


total_matrix = model_predictions[0]

for i in range(1, num_folds):
    total_matrix += model_predictions[i]

accuracy = (total_matrix[0][0] + total_matrix[1][1] + total_matrix[2][2])/79
print(features)
print(accuracy)
print(total_matrix)


# Full features list: ['count', 'cont', 'retain', 'shift', 'one_mention',
#        'multi_mention', 'flesch_reading_ease', 'difficult words', 'dale_chall_readability_score',
#        'textstat difficult words', 'arc_length', 'average_height', 'height',
#        'num_edus']
