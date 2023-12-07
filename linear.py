import numpy as np
import pandas as pd
import sklearn.linear_model

dataframe = pd.read_excel('data/TOEFL Annotation.xlsx')

dataframe['level'] = dataframe['level'].replace('low', '0').replace('medium', '1').replace('high', '2')

dataframe = dataframe.sample(frac = 1)

length = len(dataframe['level'])

# features = ['count', 'cont', 'retain', 'shift', 'one_mention', 'multi_mention']

features = ['cont', 'retain', 'shift', 'one_mention', 'multi_mention']

# X_train = dataframe[['count']][0:int(length*.8)]
X_train = dataframe[features][0:int(length*.8)]
Y_train = dataframe['level'][0:int(length*.8)]

# print(X_train)
# print(Y_train)
# X_test = dataframe[['count']][int(length*.8):-1]
X_test = dataframe[features][int(length*.8):-1]
Y_test = dataframe['level'][int(length*.8):-1]


classifier = sklearn.linear_model.LogisticRegression(multi_class='multinomial', random_state=0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
# y_pred = []
#
# for item in y_pred_raw:
#     if item <= 0.5:
#         y_pred.append(0)
#     elif item <= 1.5:
#         y_pred.append(1)
#     else:
#         y_pred.append(2)


print(y_pred)
print(Y_test)

accuracy = np.mean(y_pred == Y_test)

high = 0
true_high = 0
medium = 0
true_med = 0
low = 0
true_low = 0

for index, true in enumerate(Y_test):
    true = int(true)
    pred = int(y_pred[index])
    print(str(true) + " " + str(y_pred[index]))
    if true == 0:
        true_low += 1
    elif true == 1:
        true_med += 1
    elif true == 2:
        true_high += 1
    if true == pred:
        if true == 0:
            print("low")
            low += 1
        elif true == 1:
            print("medium")
            medium += 1
        elif true == 2:
            print("high")
            high += 1

high = high/true_high
medium = medium/true_med
low = low/true_low

print("The accuracy of the classifier is:", accuracy)
print(high)
print(medium)
print(low)