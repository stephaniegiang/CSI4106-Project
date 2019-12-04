import helper
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd

training_set, training_target, test_set, test_ids = helper.get_data()

X_train, X_test, y_train, y_test = train_test_split(training_set, training_target, test_size = 0.2, random_state=1)
train_encoded, test_encoded = helper.encode(X_train, X_test)

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
  reg_alpha=0.00006
)

folds = 5
param_comb = 5

skf = StratifiedKFold(shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgbA, param_distributions=params, n_jobs=1,cv=2 )

random_search.fit(train_encoded, y_train)
print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)