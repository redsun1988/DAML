from typing import Dict
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

class SklearnModelParameters:
    @classmethod
    def get_defalut_options(cls, model) -> Dict[str,any]:
        params = {
            #"vect": (CountVectorizer(), TfidfVectorizer()),
            #"vect__analyzer": ("word", "char", "char_wb"),
            #"vect__stop_words": ("english", None), 
            #"vect__binary": (False, True), 
            #"vect__lowercase": (False, True),
            #"vect__ngram_range": ((1,2), (1,3), (1,4)),
            #"vect__max_df": np.arange(0.85, 1.0, 0.01), 
        }
        if isinstance(model, type(LinearSVC)):
            pass
            #params["clf__penalty"] = ("l1", "l2")
            #params["clf__loss"] = ("hinge", "squared_hinge")
            #params["clf__C"] = np.arange(0.8, 1.5, 0.1)
        return cls.customize_options(params, model)
    @staticmethod
    def customize_options(params:Dict[str,any], model: any) -> Dict[str,any]:
        return params
