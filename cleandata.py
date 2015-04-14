import pandas as pd
import numpy as np
import scipy as sp

def standardize(df):

    meanFeats = df.ix[:, 'feat_1': 'feat_93'].mean()
    stdFeats =  df.ix[:, 'feat_1': 'feat_93'].std()
    df.ix[:, 'feat_1': 'feat_93'] = (df.ix[:, 'feat_1': 'feat_93'] - meanFeats) / stdFeats

    return df

def cleaneddf():

    train_df = pd.read_csv('train.csv', header=0)
    test_df = pd.read_csv('test.csv', header=0)

    # convert target column to numeric
    train_df['targetId'] = pd.factorize(train_df['target'])[0] + 1
    train_df = train_df.drop(['target'], axis=1)

    # standardize (zero mean, normalization)
    train_df = standardize(train_df)
    test_df = standardize(test_df)

    # shuffle the training set
    randomIndex = np.random.permutation(len(train_df))
    train_df = train_df.ix[randomIndex]

    return [train_df, test_df]


def logloss(act, pred):

    epsilon = 1e-15
    act = pd.get_dummies(act).values
    pred = sp.minimum(1 - epsilon, pred)
    pred = sp.maximum(epsilon, pred)
    ll = np.sum(act * sp.log(pred))
    ll = ll * -1.0/ len(act)
    return ll

def write_to_csv(id, results):
    output = np.column_stack((id, results))
    df_results = pd.DataFrame(output, columns=['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
    df_results['id'] = df_results['id'].astype('int')
    df_results.to_csv('otto_results.csv', index=False)







