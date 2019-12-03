import helper
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

training_set, training_target, test_set, test_ids = helper.get_data()

X_train, X_test, y_train, y_test = train_test_split(training_set, training_target, test_size = 0.2, random_state=1)
train_encoded, test_encoded = helper.encode(X_train, X_test)

parameters = {
  'bootstrap': [True, False],
  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
  'max_features': ['auto', 'sqrt'],
  'min_samples_leaf': [1, 2, 4],
  'min_samples_split': [2, 5, 10],
  'n_estimators': [20, 30, 40, 50, 60, 70]
}

clf = GridSearchCV(RandomForestRegressor(), parameters)
clf.fit(train_encoded, y_train)
print('score',clf.score(train_encoded, y_train))
print(clf.best_params_) 
