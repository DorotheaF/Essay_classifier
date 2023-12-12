import pandas as pd

dataframe = pd.read_excel('data/TOEFL Annotation.xlsx')

added_auto = pd.read_excel('data/word_difficulty_final.xlsx')

added_rst = pd.read_excel('data/rst_features.xlsx')

pd.set_option('display.max_columns', None)

dataframe = dataframe.merge(added_auto, how='left', on='name')
dataframe = dataframe.merge(added_rst, how='left', on='name')

print(dataframe.columns)
print(len(dataframe.index))
print(dataframe.head())

dataframe.to_excel('data/TOEFL Annotation_mine.xlsx')