import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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


parameters = {
  'bootstrap': [True, False],
  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
  'max_features': ['auto', 'sqrt'],
  'min_samples_leaf': [1, 2, 4],
  'min_samples_split': [2, 5, 10],
  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

clf = GridSearchCV(RandomForestRegressor(), parameters)
clf.fit(train_encoded, y_train)
print('score',clf.score(train_encoded, y_train))
print(clf.best_params_)