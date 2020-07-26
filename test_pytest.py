import pytest
import numpy as np
from dolly_auto_classifiers import DlTextAutoClassifier
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC


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
