import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# read the train set data 
X = pd.read_csv('data.csv')
y = X.pop("SalePrice").values
X.pop('Id')

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def replace_Na(data):
    type(data)
    clist = data.columns[data.isna().any()].tolist()
    for col in clist:
      index = data[col].first_valid_index()
      replace = 0
      if index is not None:
        first_valid_value = data[col].loc[index]
        if not is_number(str(first_valid_value)):
          replace = ""
      data[col].fillna(replace,inplace=True)

def encode(train, test, out=False):
  replace_Na(train)
  replace_Na(test)
  if out:
    train.to_csv("traindata.csv", index=False, header=True)
  ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

  ohe.fit(train)

  X_train_encoded = ohe.transform(train)
  X_test_encoded = ohe.transform(test)

  return X_train_encoded, X_test_encoded

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
train_encoded, test_encoded = encode(X_train, X_test)


params = {
        'learning_rate':[0.01,0.02,0.05,0.1,0.2,0.25,0.5,0.75,1],
        'min_child_weight': [1, 5, 10],
        'max_depth': [3, 4, 5]
        }

xgbA =xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
#folds = 5
#param_comb = 5

skf = StratifiedKFold( shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgbA, param_distributions=params,
                                   scoring='accuracy',
                                   n_jobs=4, cv=skf.split(train_encoded,y_train),
                                   verbose=3, random_state=1001 )

random_search.fit(train_encoded, y_train)
#timer(start_time) # timing ends here for "start_time" variable
print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
