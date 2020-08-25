import pandas as pd
import numpy as np
pd.set_option('max_columns',None)
import seaborn as sns
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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


clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)

clf.fit(X_train, y_train)
pred = pd.DataFrame(clf.predict(X_test))
print(mean_absolute_error(y_test, pred))


