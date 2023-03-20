import pandas as pd
from sklearn.base import clone 
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

class Models:
    classification_models = [
        DummyClassifier(),
        GaussianNB(),
        LogisticRegression(),
        MLPClassifier(),
        RandomForestClassifier(),
        SVC(),
        XGBClassifier()
    ]

    regression_models = [
        DummyRegressor(),
        LinearRegression(),
        MLPRegressor(),
        RandomForestRegressor(),
        SGDRegressor(),
        SVR(),
        XGBRegressor()
    ]
    
class AutoMLBaseline:
    """
    Get a baseline model score for either regression or classification
    """
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_task: str = "classification",
        scoring: str = None,
        n_cv: int = 5,
    ):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.model_task = model_task
        self.n_cv= n_cv
        self.model_performance = {}
    
    def _get_models(self):
        """
        Get the relevant models and set the default scoring method
        if not provided
        """
        if self.scoring is None:
            print("scoring not set - reverting to default")
            
        if self.model_task == "classification":
            return Models.classification_models
        elif self.model_task == "regression":
            return Models.regression_models
        
    def _get_cross_val_score(self, clf):
        # Run stratified K fold CV if classification
        stratified_kfold = StratifiedKFold(n_splits=self.n_cv)

        return cross_val_score(
            clf,
            self.X,
            self.y,
            cv=stratified_kfold if self.model_task == "classification" else self.n_cv,
            scoring=self.scoring,
        )
    
    def run_experiment(self):
        for model in self._get_models():
            model_ = clone(model)
            raw_scores = self._get_cross_val_score(model_)
            self.model_performance[type(model).__name__] = raw_scores
        
        df = pd.DataFrame({
            k: v.mean() for k, v in  self.model_performance.items()
        }, index=["score"]).T
        return df.sort_values("score", ascending=False)
            