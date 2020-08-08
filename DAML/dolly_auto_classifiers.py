from .abstract_classifiers import DlGroupClassifier
from .sklearn_classifiers import DlSvcTextClassifier, DlForestTextClassifier, DlLogTextClassifier
from .tensor_flow_classifiers import DlConvTextClassifier

class DlTextAutoClassifier(DlGroupClassifier):
    def _add_models(self) -> None:
        self._models.append(DlSvcTextClassifier())
        self._models.append(DlLogTextClassifier())
        self._models.append(DlForestTextClassifier())
        self._models.append(DlConvTextClassifier())
        