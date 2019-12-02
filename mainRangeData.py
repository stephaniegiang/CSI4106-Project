from math import sqrt
from math import log
import pandas as pd
import random
import datetime
import xgboost as xgb
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# read the train set data 
X = pd.read_csv('trainingset_v2.csv')
y = X.pop("SalePrice").values
X.pop('Id')

# read the train set data 
X2 = pd.read_csv('test.csv')
ids = X2.pop('Id')

def split_train_test(number_of_features=10, seed = 0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
    random.seed=seed

    randomFeatures = random.sample(list(X), number_of_features)

    X_train = X_train[randomFeatures].copy()
    X_test = X_test[randomFeatures].copy()
    
    return randomFeatures, X_train, X_test, y_train, y_test

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

def train_model(clf, X_train, y_train, epochs=3):
  scores = []
  print("Starting training...")
  for i in range(1, epochs + 1):
    print("Epoch:" + str(i) + "/" + str(epochs) + " -- " + str(datetime.datetime.now()))
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    scores.append(score)
  print("Done training.")
  return scores

def rsme_eval(clf, x, y_test, test_encoded):
  log_y_train = list(map(lambda x: log(x), y_test))
  log_pred = list(map(lambda x: log(x), clf.predict(test_encoded)))
  test_score = sqrt(mean_squared_error(log_y_train, log_pred))
  return test_score

def run_with_feature(clf, best_features, epochs, filename, seed):
  train = X[best_features].copy()
  test = X2[best_features].copy()

  print("***encoding Test Data***")
  train_encoded, test_encoded = encode(train, test)
  clf_scores = train_model(clf, train_encoded, y, epochs)

  score = rsme_eval(clf, X, y, train_encoded)

  print("\nDone training, Scores:\n", clf_scores)
  print("\npredicting and saving to",filename)

  y_predicted = clf.predict(test_encoded)
  df = pd.DataFrame(columns=['Id', 'SalePrice'])
  for i in range(len(y_predicted)):
    df.loc[i] = [ids[i],y_predicted[i]]
  df = df.astype({'Id':int})
  print(df.dtypes)

  df.to_csv(filename, index=None, header=True)
  with open("result.txt", 'a+') as a:
    a.write("features: "+str(best_features))
    a.write('\n score: '+str(score))
    a.write('\n seed: '+str(seed))
    a.write('\n')

def find_best_feat(clf):
  scores = -999
  bfeatures = []
  seed = 0
  number_of_feat = 0
  epochs = 1
  runs = 3
  start = datetime.datetime.now()
  for i in range(runs):
    num_feat = random.randint(1,76)
    
    features, train, test, y_train, y_test = split_train_test(num_feat,i)
    train_encoded, test_encoded = encode(train, test)
    train_model(clf, train_encoded, y_train, epochs)

    y_predicted = clf.predict(train_encoded[0:10])

    test_score = rsme_eval(clf, X, y_test, test_encoded)

    if test_score < scores or scores == -999:
      bfeatures = features
      scores = test_score
      seed = i
      number_of_feat = num_feat
    approx = ((((datetime.datetime.now() - start).seconds) / (i+1)) * runs) - (datetime.datetime.now() - start).seconds
    print("time left: %s | %s/%s | best score: %s"%(str(datetime.timedelta(seconds=approx)),i+1,runs,scores), end='\r')
  
  print("\n\n best features:", bfeatures)
  print('best score:',scores)
  print('seed:',seed)

  run_with_feature(clf, bfeatures,epochs,"predictions.csv",seed)

def get_random_forest_v1():
  return RandomForestRegressor(n_estimators=20)

def get_random_forest_v2():
  return RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

def get_decision_tree_v1():
  return DecisionTreeRegressor()

def get_decision_tree_v2():
  return DecisionTreeRegressor()

def get_xgb_v1():
  return xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
    max_depth=3, min_child_weight=0,
    gamma=0, subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear', nthread=-1,
    scale_pos_weight=1, seed=27,
    reg_alpha=0.00006)

def get_xgb_v2():
  return xgb.XGBRegressor(learning_rate=0.02,n_estimators=3460,
    max_depth=3, min_child_weight=0,
    gamma=0, subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear', nthread=-1,
    scale_pos_weight=1, seed=27,
    reg_alpha=0.00006)

find_best_feat(get_random_forest_v1())

