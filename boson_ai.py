import pandas as pd
import numpy as np
pd.set_option('max_columns',None)
import seaborn as sns
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import eli5
from eli5.sklearn import PermutationImportance

train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\higgs boson ml challenge\training.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\higgs boson ml challenge\test.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\higgs boson ml challenge\random_submission.csv')

dict = {'s':0,'b':1}
train['Label'] = train['Label'].map(dict)
y = train.Label
train = train.drop(['Label', 'Weight'], axis = 1)


X = train
del train

sns.set(style="darkgrid")
ax = sns.barplot(x = y.value_counts().index, y = y.value_counts())


X_cols, test_cols = X.columns, test.columns

X = X.replace(-999.000,np.nan)
test = test.replace(-999.000,np.nan)

imp = SimpleImputer(missing_values = np.nan, strategy = "mean")
imp.fit(X)
X = pd.DataFrame(imp.transform(X))

imp.fit(test)
test = pd.DataFrame(imp.transform(test))
X.columns, test.columns = X_cols, test_cols

X.EventId = X.EventId.astype('int32')
test.EventId = test.EventId.astype('int32')

X.set_index('EventId', inplace = True)
test.set_index('EventId', inplace = True)

X = pd.DataFrame(normalize(X))
test = pd.DataFrame(normalize(test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2020)

estimators_to_test = [50, 100, 150, 200]

def xg_tester(estimators_to_test, X_train, X_test, y_train, y_test):
    clf = xgb.XGBClassifier(n_estimators = estimators_to_test, random_state = 2020)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return(mae)


for i in estimators_to_test:
    print(f"n_estimators: {i}, mae: {xg_tester(i, X_train, X_test, y_train, y_test)}")
    

X_list = X_test.columns.tolist()

clf = xgb.XGBClassifier(n_estimators = 150, random_state = 2020)
clf.fit(X_train, y_train)
perm = PermutationImportance(clf, random_state = 2010)
perm.fit(X_test, y_test)
# Store feature weights in an object
html_obj = eli5.show_weights(perm, feature_names = X_list)
# Write html object to a file (adjust file path; Windows path is used here)
with open(r'C:\Users\lukem\Desktop\Github AI Projects\Higgs-Boson-machine-learning-challenge\boson-importance.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))
    
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"logistic regression, mae: {mae}")

def rf_tester(estimators_to_test, X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators = estimators_to_test, random_state = 2020)
    rf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return(mae)

for i in estimators_to_test:
    print(f"n_estimators: {i}, mae: {rf_tester(i, X_train, X_test, y_train, y_test)}")