import pandas as pd
from sklearn.feature_selection import RFECV
import sklearn.naive_bayes
from sklearn.model_selection import StratifiedKFold
import sklearn.svm

dataframe = pd.read_excel('data/TOEFL Annotation_1.xlsx')
dataframe['level'] = dataframe['level'].replace('low', '0').replace('medium', '1').replace('high', '2')

features = ['count', 'cont', 'retain', 'shift', 'one_mention',
            'multi_mention', 'flesch_reading_ease', 'difficult words', 'dale_chall_readability_score']

X = dataframe[features]
y = dataframe['level']

min_features_to_select = 1  # Minimum number of features to consider
clf = sklearn.svm.LinearSVC(dual="auto")
cv = StratifiedKFold(5, shuffle=True)
print(cv)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")