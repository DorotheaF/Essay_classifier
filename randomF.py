import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier

dataframe = pd.read_excel('data/TOEFL Annotation.xlsx')

dataframe['level'] = dataframe['level'].replace('low', '0').replace('medium', '1').replace('high', '2')

dataframe = dataframe.sample(frac = 1, random_state=0)

split_num = 68
features = ['count', 'cont', 'retain', 'shift', 'one_mention', 'multi_mention']

X_train = dataframe[features][0:split_num]
Y_train = dataframe['level'][0:split_num]

X_test = dataframe[features][split_num:-1]
Y_test = dataframe['level'][split_num:-1]

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

accuracy = np.mean(y_pred == Y_test)
matrix = sklearn.metrics.confusion_matrix(Y_test, y_pred)

print("The accuracy of the classifier is:", accuracy)
print(matrix)