from pathlib import Path
import scanpy as sc
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression, r_regression
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import matthews_corrcoef
from sklearn.multioutput import MultiOutputRegressor


FTR_SELS = {"skb": SelectKBest, "sfm": SelectFromModel}
FTR_ESTIMATORS = {"f": f_regression, "r": r_regression}

REGRESSORS = {"knr": KNeighborsRegressor,
              "rfr": RandomForestRegressor,
              "svr": SVR,
              "dtr": DecisionTreeRegressor,
              "rnr": RadiusNeighborsRegressor,
              "sgd": SGDRegressor,
              "gbr": GradientBoostingRegressor,
              "lr": LinearRegression}

# xgboost? voting?


class Regressor:
    def __init__(self,
                 reg_name,
                 fs_name,
                 fs_est):
        self.reg = REGRESSORS[reg_name]()
        self.fs = FTR_SELS[fs_name](FTR_ESTIMATORS[fs_est])
        self.pipeline = Pipeline([("ftr", self.fs),
                                  ("regr", self.reg)])
        self.multi_output = MultiOutputRegressor(self.pipeline)
        self.cv = None

    def cross_validation(self, X, y, param_dist,
                         cv=5,
                         scoring="r2",
                         n_iter=10,
                         n_jobs=1,
                         verbose=1):
        print("Start training model")
        print(f"scoring method: {scoring}")
        print(f"cv: {cv}")
        print(f"n_iter: {n_iter}")

        self.cv = RandomizedSearchCV(self.multi_output,
                                     param_distributions=param_dist,
                                     cv=cv,
                                     n_jobs=n_jobs,
                                     scoring=scoring,
                                     n_iter=n_iter,
                                     verbose=verbose)
        self.cv.fit(X, y)
        print("Training finished.")
        print("Best score:", self.cv.best_score_)
        print("Best params:", self.cv.best_params_)

    def predict(self, X_test):
        return self.cv.predict(X_test)

    def save_iters(self, path):
        pd.DataFrame(self.cv.cv_results_).to_csv(path)

    def save_model(self, path):
        joblib.dump(self.cv.best_estimator_, path)

    @classmethod
    def load_model(cls, path):
        return joblib.load(path)