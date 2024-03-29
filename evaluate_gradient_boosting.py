import helper
import xgboost as xgb

def get_v1():
  return xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
    max_depth=3, min_child_weight=0,
    gamma=0, subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear', nthread=-1,
    scale_pos_weight=1, seed=27,
    reg_alpha=0.00006)

def get_v2():
  return xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=3460,
             n_jobs=1, nthread=-1, objective='reg:linear', random_state=0,
             reg_alpha=6e-05, reg_lambda=1, scale_pos_weight=1, seed=27,
             silent=None, subsample=0.7, verbosity=1)

training_set, training_target, test_set, test_ids = helper.get_data()

feature_set_1 = ['Neighborhood', '3SsnPorch', 'HeatingQC', 'LotArea', 'SaleType', 'MasVnrType', 'LotShape', 'OpenPorchSF', 'FullBath', 'BsmtFinSF1', 'MSZoning', 'TotalBsmtSF', 'HalfBath', 'GarageType', 'BsmtFinSF2', 'BsmtUnfSF', 'RoofMatl', 'MSSubClass', 'Condition1', 'MasVnrArea', 'Exterior1st', 'KitchenAbvGr', 'GarageYrBlt', 'Foundation', 'OverallCond', 'BsmtHalfBath', 'EnclosedPorch', 'BsmtFinType2', 'BldgType', 'FireplaceQu', 'LotFrontage', 'LandSlope', 'GarageFinish', 'BedroomAbvGr', 'YearBuilt', 'MiscVal', '1stFlrSF', 'OverallQual', 'MoSold']
feature_set_2 = ['OpenPorchSF', 'GarageYrBlt', '1stFlrSF', 'SaleType', 'LandContour', 'HouseStyle', 'BsmtFinType1', 'WoodDeckSF', 'LotShape', 'YrSold', 'BsmtFinSF1', 'BsmtQual', 'YearBuilt', 'BsmtCond', 'GarageArea', 'RoofStyle', 'GrLivArea', 'Electrical', 'TotalBsmtSF', 'BedroomAbvGr', 'KitchenQual', 'PavedDrive', 'BsmtExposure', 'Fireplaces', 'YearRemodAdd', 'Heating', 'BldgType', 'OverallCond', 'LotConfig', 'EnclosedPorch', 'Functional', 'OverallQual', 'ExterCond', 'LandSlope', 'GarageType', '2ndFlrSF', 'MiscVal', 'MSSubClass', 'BsmtHalfBath', 'LotArea', 'MSZoning', 'LotFrontage', 'Neighborhood', 'LowQualFinSF', 'Foundation', '3SsnPorch', 'FullBath', 'BsmtFinSF2', 'PoolArea', 'BsmtFinType2', 'RoofMatl', 'HalfBath', 'GarageCond', 'HeatingQC', 'GarageFinish', 'MasVnrArea', 'KitchenAbvGr', 'BsmtFullBath', 'GarageQual', 'Utilities', 'TotRmsAbvGrd']
feature_set_3 = ['ExterCond', 'BsmtUnfSF', 'LotArea', 'Condition1', 'Condition2', 'LotShape', 'BsmtHalfBath', 'CentralAir', 'GarageArea', 'Functional', 'WoodDeckSF', 'FullBath', 'BsmtFullBath', 'Neighborhood', 'EnclosedPorch', 'Electrical', 'GarageFinish', 'GarageType', 'Foundation', 'YearBuilt', 'BsmtFinSF1', 'BldgType', 'GarageYrBlt', 'MasVnrArea', 'Street', 'SaleCondition', '2ndFlrSF', 'LandContour', 'RoofStyle', 'MasVnrType', 'Fireplaces', 'YrSold', 'GarageCars', 'BsmtFinType1', 'OpenPorchSF', 'TotalBsmtSF', 'Exterior2nd', '1stFlrSF', 'OverallCond', 'HeatingQC', 'MoSold', 'Heating', 'KitchenQual', 'LotFrontage', 'RoofMatl', 'BedroomAbvGr', 'LandSlope', 'OverallQual', 'FireplaceQu', '3SsnPorch', 'SaleType', 'MiscVal', 'BsmtFinType2', 'ScreenPorch', 'Exterior1st', 'Utilities', 'GarageQual', 'Fence', 'BsmtCond', 'GarageCond', 'HalfBath', 'HouseStyle', 'KitchenAbvGr', 'ExterQual', 'YearRemodAdd', 'BsmtQual', 'TotRmsAbvGrd', 'LotConfig', 'GrLivArea', 'MSZoning', 'PoolArea', 'BsmtFinSF2', 'LowQualFinSF']

xgb_v1_1 = get_v1()
xgb_v1_2 = get_v1()
xgb_v1_3 = get_v1()
xgb_v2_1 = get_v2()
xgb_v2_2 = get_v2()
xgb_v2_3 = get_v2()

train1, test1, y_train1, y_test1 = helper.evaluate_split_train_test(training_set, training_target, feature_set_1)
train2, test2, y_train2, y_test2 = helper.evaluate_split_train_test(training_set, training_target, feature_set_2)
train3, test3, y_train3, y_test3 = helper.evaluate_split_train_test(training_set, training_target, feature_set_3)

train_encoded1, test_encoded1 = helper.encode(train1, test1)
train_encoded2, test_encoded2 = helper.encode(train2, test2)
train_encoded3, test_encoded3 = helper.encode(train3, test3)

helper.train_model(xgb_v1_1, train_encoded1, y_train1, 1)
helper.train_model(xgb_v1_2, train_encoded2, y_train2, 1)
helper.train_model(xgb_v1_3, train_encoded3, y_train3, 1)
helper.train_model(xgb_v2_1, train_encoded1, y_train1, 1)
helper.train_model(xgb_v2_2, train_encoded2, y_train2, 1)
helper.train_model(xgb_v2_3, train_encoded3, y_train3, 1)

rsme = []
rsme.append(helper.rsme_eval(xgb_v1_1, y_test1, test_encoded1))
rsme.append(helper.rsme_eval(xgb_v1_2, y_test2, test_encoded2))
rsme.append(helper.rsme_eval(xgb_v1_3, y_test3, test_encoded3))
rsme.append(helper.rsme_eval(xgb_v2_1, y_test1, test_encoded1))
rsme.append(helper.rsme_eval(xgb_v2_2, y_test2, test_encoded2))
rsme.append(helper.rsme_eval(xgb_v2_3, y_test3, test_encoded3))

helper.run_predictions(get_v1(), feature_set_1, 1,"xgb_v1_f1_predictions.csv", training_set, test_set, training_target, test_ids)
helper.run_predictions(get_v1(), feature_set_2, 1,"xgb_v1_f2_predictions.csv", training_set, test_set, training_target, test_ids)
helper.run_predictions(get_v1(), feature_set_2, 1,"xgb_v1_f3_predictions.csv", training_set, test_set, training_target, test_ids)
helper.run_predictions(get_v2(), feature_set_1, 1,"xgb_v2_f1_predictions.csv", training_set, test_set, training_target, test_ids)
helper.run_predictions(get_v2(), feature_set_2, 1,"xgb_v2_f2_predictions.csv", training_set, test_set, training_target, test_ids)
helper.run_predictions(get_v2(), feature_set_2, 1,"xgb_v2_f3_predictions.csv", training_set, test_set, training_target, test_ids)

print(rsme)
