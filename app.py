# Load libraries -------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import optuna
from sklearn.metrics import root_mean_squared_error
from flaml import AutoML

# Load and process data ------------------------------------------------------------------------------------------------

solTrainX = pd.read_csv("data/solubility/solTrainX.csv")
solTrainY = pd.read_csv("data/solubility/solTrainY.csv").values.ravel()
solTestX = pd.read_csv("data/solubility/solTestX.csv")
solTestY = pd.read_csv("data/solubility/solTestY.csv").values.ravel()

# Create preprocessing pipeline
# Normalize data, set missing values to 0, and remove features with 0 variance.
estimators = [('center_and_scale', preprocessing.StandardScaler()),
              ('impute', SimpleImputer(strategy='median')),
              ('remove_zv', VarianceThreshold())]

pipe = Pipeline(estimators)
pipe.fit(solTrainX)
train_x_trans = pipe.transform(solTrainX)
test_x_trans = pipe.transform(solTestX)

# XGBoost model with default hyperparameters ---------------------------------------------------------------------------

# Standard model without tuning
xgb_mod = xgb.XGBRegressor(objective="reg:squarederror")

xgb_mod_default_cv = cross_val_score(estimator=xgb_mod,
                                     X=train_x_trans,
                                     y=solTrainY,
                                     cv=5,
                                     scoring='neg_root_mean_squared_error',
                                     verbose=2)

xgb_mod_default_cv_score = np.mean(xgb_mod_default_cv)
print(xgb_mod_default_cv_score)


# Automatic model tuning using Optuna ----------------------------------------------------------------------------------

# Set up tuning trial
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    model = xgb.XGBRegressor()
    return cross_val_score(model, X=train_x_trans, y=solTrainY,
                           scoring='neg_root_mean_squared_error',
                           cv=5).mean()


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(lambda trial: objective(trial), n_trials=250)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))

# Get CV score of the best model
xgb_mod_upd = xgb.XGBRegressor(objective='reg:squarederror',
                               n_estimators=1000,
                               verbosity=0,
                               learning_rate=study.best_trial.params['learning_rate'],
                               max_depth=study.best_trial.params['max_depth'],
                               subsample=study.best_trial.params['subsample'],
                               colsample_bytree=study.best_trial.params['colsample_bytree'],
                               min_child_weight=study.best_trial.params['min_child_weight'])

xgb_mod_tuned_cv = cross_val_score(estimator=xgb_mod_upd,
                                   X=train_x_trans,
                                   y=solTrainY,
                                   cv=5,
                                   scoring='neg_root_mean_squared_error',
                                   verbose=2)

xgb_mod_optuna_cv_score = np.mean(xgb_mod_tuned_cv)
print(xgb_mod_optuna_cv_score)

'''
# Fit final model.
final_model = xgb_mod_upd.fit(X=train_x_trans, y=solTrainY)
# Predict test set values
final_predictions = final_model.predict(X=test_x_trans)

root_mean_squared_error(solTestY, final_predictions)
# This is the best model so far with an RMSE of 0.593.
'''

# Manually tune XGB models ----------------------------------------------------

xgb_grid = {
    "n_estimators": [1000],
    # "verbosity": 0,
    # "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]
    "learning_rate": [0.1],
    "max_depth": [3],
    "subsample": [0.6],
    "colsample_bytree": [0.9],
    "min_child_weight": [1],
    "lambda": [7],
    "alpha": [2]
}

xgb_mod_manual_grid = xgb.XGBRegressor(objective="reg:squarederror")
xgb_manual_grid_mod = GridSearchCV(estimator=xgb_mod_manual_grid, param_grid=xgb_grid,
                                   scoring='neg_root_mean_squared_error', cv=5, verbose=2)

xgb_manual_grid_mod.fit(train_x_trans, solTrainY)
grid_search_result = pd.DataFrame(xgb_manual_grid_mod.cv_results_)

print(grid_search_result['mean_test_score'])

'''
final_model = xgb_manual_grid_mod.fit(X=train_x_trans, y=solTrainY)
# Predict test set values
final_predictions = final_model.predict(X=test_x_trans)

root_mean_squared_error(solTestY, final_predictions)
# 0.598. Good final score. It's essentially the same as the optuna trained one.
# Just a tiny bit worse than the FLAML model.
'''

# AutoML using FLAML ---------------------------------------------------------------------------------------------------

automl = AutoML()
automl.fit(train_x_trans, solTrainY, task="regression", time_budget=1800)

# Print best model
print(automl.model.estimator)

automl_cv_score = cross_val_score(estimator=automl.model,
                                  X=train_x_trans,
                                  y=solTrainY,
                                  cv=5,
                                  scoring='neg_root_mean_squared_error',
                                  verbose=2)

np.mean(automl_cv_score)

# Predict the test set
pred = automl.predict(test_x_trans)
root_mean_squared_error(test_y, pred)

# Test set error: 0.5847
# This is slightly better than my tuned xgb model from earlier today.
