import helper
import sklearn
import xgboost as xgb
import random
import datetime

def get_clf():
  return xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
    max_depth=3, min_child_weight=0,
    gamma=0, subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear', nthread=-1,
    scale_pos_weight=1, seed=27,
    reg_alpha=0.00006)

training_set, training_target, test_set, test_ids = helper.get_data()

scores = -999
bfeatures = []
seed = 0
number_of_feat = 0
epochs = 1
runs = 200
start = datetime.datetime.now()

for i in range(runs):
  clf = get_clf()
  num_feat = random.randint(1,76)
  
  features, train, test, y_train, y_test = helper.split_train_test(training_set, training_target, num_feat,i, )
  train_encoded, test_encoded = helper.encode(train, test)
  helper.train_model(clf, train_encoded, y_train, epochs)

  y_predicted = clf.predict(train_encoded[0:10])

  test_score = helper.rsme_eval(clf, training_set, y_test, test_encoded)

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
