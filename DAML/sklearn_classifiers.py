from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from .abstract_classifiers import DlSingleClassifier
from .sklearn_parameters import SklearnModelParameters

class DlSklearClassifier(DlSingleClassifier):
    def __init__(self) -> None:
        super().__init__()
        self._model = GridSearchCV(
            self._model,
            SklearnModelParameters.get_defalut_options(self._model),
            n_jobs=-1, verbose=1)

class DlSklearnTextClassifier(DlSingleClassifier):
    def create_model(self):
        model = super().create_model()
        return Pipeline([
            ("vect", CountVectorizer()),
            ("clf", model),
            ])

class DlSvcClassifier(DlSingleClassifier):
    def create_model(self) -> any:
        return LinearSVC(random_state=42)
class DlLogClassifier(DlSingleClassifier):
    def create_model(self) -> any:
        return LogisticRegression(random_state=42)
class DlForestClassifier(DlSingleClassifier):
    def create_model(self) -> any:
        return RandomForestClassifier(random_state=42)

class DlSvcTextClassifier(DlSklearClassifier, DlSklearnTextClassifier, DlSvcClassifier):
    pass
class DlLogTextClassifier(DlSklearClassifier, DlSklearnTextClassifier, DlLogClassifier):
    pass
class DlForestTextClassifier(DlSklearClassifier, DlSklearnTextClassifier, DlForestClassifier):
    pass