import numpy as np
import pandas as pd
import sklearn.naive_bayes

dataframe = pd.read_excel('data/TOEFL Annotation.xlsx')
print(dataframe.columns)

dataframe['level'] = dataframe['level'].replace('low', '0').replace('medium', '1').replace('high', '2')

length = len(dataframe['level'])

X_train = dataframe[['cont', 'retain', 'shift', 'one_mention', 'multi_mention']][0:int(length*.8)]
Y_train = dataframe['level'][0:int(length*.8)]

# print(X_train)
# print(Y_train)

X_test = dataframe[['cont', 'retain', 'shift', 'one_mention', 'multi_mention']][int(length*.8):-1]
Y_test = dataframe['level'][int(length*.8):-1]

# print(X_test)
print(Y_test)


classifier = sklearn.naive_bayes.GaussianNB()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

print(y_pred)

accuracy = np.mean(y_pred == Y_test)

print("The accuracy of the classifier is:", accuracy)