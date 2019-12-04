from math import sqrt
from math import log
import pandas as pd
import random
import datetime
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_data():
  # read the train set data 
  training_set = pd.read_csv('trainingset_v2.csv')
  training_target = training_set.pop("SalePrice").values
  training_set.pop('Id')

  # read the test set data 
  test_set = pd.read_csv('test.csv')
  test_ids = test_set.pop('Id')

  return training_set, training_target, test_set, test_ids

def evaluate_split_train_test(training_set, training_target, features):
  X_train, X_test, y_train, y_test = train_test_split(training_set, training_target, test_size = 0.2, random_state=1)

  X_train = X_train[features].copy()
  X_test = X_test[features].copy()
  
  return X_train, X_test, y_train, y_test

# used for finding best features. grabs a random set of features to use. 
def split_train_test(training_set, training_target, number_of_features=10, seed = 0):
  X_train, X_test, y_train, y_test = train_test_split(training_set, training_target, test_size = 0.2, random_state=1)
  random.seed=seed

  randomFeatures = random.sample(list(training_set), number_of_features)

  X_train = X_train[randomFeatures].copy()
  X_test = X_test[randomFeatures].copy()
  
  return randomFeatures, X_train, X_test, y_train, y_test

# checks if the value is a numnber
def is_number(s):
  """ Returns True is string is a number. """
  try:
    float(s)
    return True
  except ValueError:
    return False
    
# replaces NA values found in our data set
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

# uses one hot encoder to encode our data set 
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

# does root squared mean error evaluation 
def rsme_eval(clf, y_test, test_encoded):
  log_y_train = list(map(lambda x: log(x), y_test))
  log_pred = list(map(lambda x: log(x), clf.predict(test_encoded)))
  test_score = sqrt(mean_squared_error(log_y_train, log_pred))
  return test_score

def run_predictions(clf, features, epochs, filename, train, test, training_target, test_ids):
  train = train[features].copy()
  test = test[features].copy()

  train_encoded, test_encoded = encode(train, test)
  train_model(clf, train_encoded, training_target, epochs)

  y_predicted = clf.predict(test_encoded)

  print("Predicting and saving to ",filename)

  y_predicted = clf.predict(test_encoded)
  df = pd.DataFrame(columns=['Id', 'SalePrice'])
  for i in range(len(y_predicted)):
    df.loc[i] = [test_ids[i],y_predicted[i]]
  df = df.astype({'Id':int})
  print(df.dtypes)

  df.to_csv(filename, index=None, header=True)