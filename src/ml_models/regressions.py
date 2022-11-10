from pathlib import Path
import scanpy as sc
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

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

try:
    from cuml.ensemble import RandomForestRegressor as rfr_cuml
    from cuml.svm import SVR as svr_cuml
    from cuml.neighbors import KNeighborsRegressor as knr_cuml
    from cuml import Ridge as ridge_cuml

    REGRESSORS_CUML = {"rfr": rfr_cuml,
                       "svr": svr_cuml,
                       "knr": knr_cuml,
                       "ridge": ridge_cuml,
                       }

except ModuleNotFoundError as e:
    print(e)
    print("Use sklearn instead")

    REGRESSORS_CUML = {"knr": KNeighborsRegressor,
              "rfr": RandomForestRegressor,
              "svr": SVR,
              "dtr": DecisionTreeRegressor,
              "rnr": RadiusNeighborsRegressor,
              "gbr": GradientBoostingRegressor}

from dask_ml.model_selection import RandomizedSearchCV as dask_rscv

FTR_SELS = {"skb": SelectKBest, "sfm": SelectFromModel}
FTR_ESTIMATORS = {"f": f_regression, "r": r_regression}

REGRESSORS = {"knr": KNeighborsRegressor,
              "rfr": RandomForestRegressor,
              "svr": SVR,
              "dtr": DecisionTreeRegressor,
              "rnr": RadiusNeighborsRegressor,
              "gbr": GradientBoostingRegressor}


# xgboost? voting?


class Regressor:
    def __init__(self,
                 reg_name,
                 #fs_name,
                 #fs_est,
                 use_cuml=False):
        self._use_gpu = use_cuml
        if use_cuml:
            self.reg = REGRESSORS_CUML[reg_name]()
        else:
            self.reg = REGRESSORS[reg_name]()
            self.pipeline = Pipeline([("regr", self.reg)])
            self.multi_output = MultiOutputRegressor(self.pipeline)
        self.cv = None
        self._best_score = 0
        self._best_scores = []
        self._best_params = None

    @property
    def best_score(self):
        return np.mean(self._best_scores)

    def partial_cv(self, X, y_vec,
                   y_index,
                   param_dist,
                   n_cv=5,
                   scoring="r2",
                   n_iter=10,
                   n_jobs=1,
                   verbose=1
                   ):
        print(f"Start training model (target {y_index})")
        print(f"scoring method: {scoring}")
        print(f"cv: {n_cv}")
        print(f"n_iter: {n_iter}")
        print(f"Using GPU: {self._use_gpu}")
        if self.cv is None:
            self.cv = []
        cv = RandomizedSearchCV(self.reg,
                               param_distributions=param_dist,
                               cv=n_cv,
                               n_jobs=1,
                               scoring=scoring,
                               n_iter=n_iter,
                               verbose=verbose)
        self.cv.append(cv)
        self.cv[-1].fit(X, y_vec)
        self._best_scores.append(self.cv[-1].best_score_)


    def cross_validation(self,
                         X, y,
                         param_dist,
                         n_cv=5,
                         scoring="r2",
                         n_iter=10,
                         n_jobs=1,
                         verbose=1):
        print("Start training model")
        print(f"scoring method: {scoring}")
        print(f"cv: {n_cv}")
        print(f"n_iter: {n_iter}")
        print(f"Using GPU: {self._use_gpu}")
        assert X.shape[0] == y.shape[0]
        if self._use_gpu:
            self.cv = []
            best_scores = []
            for i in tqdm(range(y.shape[1])):
                cv = RandomizedSearchCV(self.reg,
                                        param_distributions=param_dist,
                                        cv=n_cv,
                                        n_jobs=1,
                                        scoring=scoring,
                                        n_iter=n_iter,
                                        verbose=verbose)
                self.cv.append(cv)
                self.cv[i].fit(X, y[:, i])
                best_scores.append(cv.best_score_)
            self._best_scores = best_scores
        else:
            self.cv = RandomizedSearchCV(self.multi_output,
                                         param_distributions=param_dist,
                                         cv=n_cv,
                                         n_jobs=n_jobs,
                                         scoring=scoring,
                                         n_iter=n_iter,
                                         verbose=verbose)
            self.cv.fit(X, y)
            self._best_scores.append(self.cv.best_score_)
        print("Training finished.")
        print("Best score:", self.best_score)

    def predict(self, X_test):
        if self._use_gpu:
            pred_ys = []
            for cv in self.cv:
                pred_y = cv.predict(X_test)
                pred_y = pred_y.reshape(pred_y.shape[0], 1)
                pred_ys.append(pred_y)
            return np.concatenate(pred_ys, axis=1)
        return self.cv.predict(X_test)

    def save_iters(self, path):
        if self._use_gpu:
            dfs = []
            for i, cv in enumerate(self.cv):
                df = pd.DataFrame(cv.cv_results_)
                df["target_index"] = i
                dfs.append(df)
            pd.concat(dfs, axis=0).to_csv(path)
        else:
            pd.DataFrame(self.cv.cv_results_).to_csv(path)

    def save_model(self, path):
        if self._use_gpu:
            for i, cv in enumerate(self.cv):
                joblib.dump(cv.best_estimator_, path / f"best_model_{i}.joblib")
        else:
            joblib.dump(self.cv.best_estimator_, path)

    @classmethod
    def load_model(cls, path):
        return joblib.load(path)