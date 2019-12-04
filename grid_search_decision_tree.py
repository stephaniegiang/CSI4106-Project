import helper
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

training_set, training_target, test_set, test_ids = helper.get_data()

X_train, X_test, y_train, y_test = train_test_split(training_set, training_target, test_size = 0.2, random_state=1)
train_encoded, test_encoded = helper.encode(X_train, X_test)

parameters = {
  'criterion': ['mse', 'friedman_mse', 'mae'],
  'splitter': ['best', 'random'],
  'max_depth': [10, None],
  'max_features': ['auto', 'sqrt', 'log2', None],
  'presort': [True, False]
}

clf = GridSearchCV(DecisionTreeRegressor(), parameters)
clf.fit(train_encoded, y_train)
print('score',clf.score(train_encoded, y_train))
print(clf.best_params_) 