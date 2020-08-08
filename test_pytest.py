import pytest
import numpy as np
from DAML.dolly_auto_classifiers import DlTextAutoClassifier
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
import pandas as pd
from DAML.data_balanser import DataBalanser

@pytest.fixture
def clfr_args():
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    x, y = newsgroups_train.data, newsgroups_train.target
    clfr = DlTextAutoClassifier()
    clfr.fit(x, y)
    return clfr, x, y

def test_score_DlClassifier(clfr_args):
    clfr, x, y = clfr_args[0], clfr_args[1], clfr_args[2]
    assert clfr.score(x, y) > 0.5

def test_predict_count_DlClassifier(clfr_args):
    clfr, x = clfr_args[0], clfr_args[1]
    assert len(clfr.predict(x)) == len(x)

def test_predict_range_DlClassifier(clfr_args):
    clfr, x, y = clfr_args[0], clfr_args[1], clfr_args[2]
    assert all((answer in range(len(clfr._classes)) for answer in clfr.predict(x)))

def test_predict_valuse_DlClassifier(clfr_args):
    clfr, x, y = clfr_args[0], clfr_args[1], clfr_args[2]
    assert any(p1 == p2 for p1, p2 in zip(clfr.predict(x), clfr._models[0].predict(x)))


def test_data_balanser():
    data_set = pd.DataFrame(
        data=[(1), (1), (1), (1), (0), (0)], 
        columns=["target"])
    data_balanser = DataBalanser(data_set)
    data_balanser.stategy = "downsampling"
    data_balanser.target_field = "target"
    assert len(data_balanser.target) == 4