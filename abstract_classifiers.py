"""
In this module is a test enviroment to built the library
"""
from typing import List, Tuple, Dict
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
"""
The bacis classifier for the library
"""
class DlBaseClassifier:
    def fit(self, x: List[str], y: List[int]) -> None:
        raise NotImplementedError("Implement the method for %s" % self.__class__.__name__)
    def score(self, x: List[str], y: List[float]) -> float:
        raise NotImplementedError("Implement the method for %s" % self.__class__.__name__)
    def predict(self, x: List[str]) -> List[int]:
        raise NotImplementedError("Implement the method for %s" % self.__class__.__name__)
class DlSingleClassifier(DlBaseClassifier):
    def __init__(self) -> None:
        self._model = self.create_model()
    def create_model(self) -> any:
        raise NotImplementedError("Implement the method for %s" % self.__class__.__name__)
    def fit(self, x: List[str], y: List[int]) -> None:
        self._model.fit(x, y)
    def score(self, x: List[str], y: List[float]) -> float:
        return self._model.score(x, y)
    def predict(self, x: List[str]) -> List[int]:
        return self._model.predict(x)
class DlGroupClassifier(DlBaseClassifier):
    def __init__(self) -> None:
        self._models: List = []
        self.one_hot_encoder: OneHotEncoder = OneHotEncoder(sparse=False)
        self._add_models()
    def _add_models(self) -> None:
        pass
    def fit(self, x: List[str], y: List[int]) -> None:
        self._classes: List = np.unique(y)
        self.one_hot_encoder.fit([[value] for value in y])
        for model in self._models:
            model.fit(x, y)
    def score(self, x: List[str], y: List[float]) -> float:
        return np.mean([model.score(x, y) for model in self._models])
    def predict(self, x: List[str]) -> List[int]:
        base = np.zeros((len(x), len(self._classes)))
        for model in self._models:
            base = np.add(base, self.one_hot_encoder.transform(
                model.predict(x).reshape(-1, 1)))
        return [np.argmax(row) for row in base]
