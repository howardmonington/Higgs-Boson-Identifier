import pandas as pd
import numpy as np
pd.set_option('max_columns',None)
import seaborn as sns
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBClassifier

train = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\higgs boson ml challenge\training.csv')
test = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\higgs boson ml challenge\test.csv')
sample_submission = pd.read_csv(r'C:\Users\lukem\Desktop\Github AI Projects\Data for ai competitions\higgs boson ml challenge\random_submission.csv')

y = train.Label
train = train.drop(['Label', 'Weight'], axis = 1)
#train = train.drop('Weight', axis = 1)

train_cols, test_cols = train.columns, test.columns


X = train
del train

X.set_index('EventId', inplace = True)
test.set_index('EventId', inplace = True)

X = X.replace(-999.000,np.nan)
test = test.replace(-999.000,np.nan)

imp = SimpleImputer(missing_values = np.nan, strategy = "mean")
imp.fit(X)
X = pd.DataFrame(imp.transform(X))

imp.fit(test)
test = pd.DataFrame(imp.transform(test))

X.columns = train_cols
test.columns = test_cols

sns.set(style="darkgrid")
ax = sns.barplot(x = y.value_counts().index, y = y.value_counts())

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)

clf.fit(X, y)

pred = pd.DataFrame(clf.predict(test))


