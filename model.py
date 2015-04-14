import cleandata
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

reload(cleandata)
train_df, test_df = cleandata.cleaneddf()


# split data for fitting
train_data = train_df.values
test_data = test_df.values
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data[::, 1:94], train_data[::, -1], test_size=0.1, random_state=0)


clf = RandomForestClassifier(n_estimators=400)
clf.fit(X_train, y_train)
val_pred = clf.predict_proba(X_test)
logloss = cleandata.logloss(y_test, val_pred)
print 'logloss of validation set:', logloss

clf.fit(train_data[::, 1:94], train_data[::, -1])
test_label = clf.predict_proba(test_data[::, 1:94])
cleandata.write_to_csv(test_data[::, 0], test_label)