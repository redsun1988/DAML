from typing import List
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .abstract_classifiers import DlSingleClassifier
from tensorflow.keras.utils import to_categorical

class DlTensorTextClassifier(DlSingleClassifier):
    def __init__(self) -> None:
        self.vocab_size = 1000
        self.embedding_dim = 124
        self.max_length = 120
        self.trunc_type='post'
        self.padding_type='post'
        self.oov_tok = "<OOV>"
        self.num_epochs: int = 50
        self.tokenizer = self._create_tokenizer()

    def _create_tokenizer(self):
        raise NotImplementedError("Implement this method for %s" % self.__class__.__name__)

    def _convert_text(self, x) -> List[List[str]]:
        raise NotImplementedError("Implement this method for %s" % self.__class__.__name__)

    def fit(self, x: List[str], y:List[int]):
        self._classes: List = np.unique(y)
        self._model = self.create_model()
        self.tokenizer.fit_on_texts(x)
        padded_x = self._convert_text(x)

        if len(self._classes) > 2:
            loss = "categorical_crossentropy"
            y = to_categorical(y, len(self._classes))
        else:
            loss = "binary_crossentropy"
        self._model.compile(loss=loss,optimizer='adam',metrics=['accuracy'])
        self._model.fit(
            padded_x, 
            y, 
            epochs=self.num_epochs, 
            verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping()])
            
    def predict(self, x: List[str]) -> List[int]:
        padded_text = self._convert_text(x)
        if len(self._classes) > 2:
            return np.argmax(self._model.predict(padded_text), axis=-1)
        else:
            return (self._model.predict(padded_text) > 0.5).astype("int32")

    def score(self, x: List[str], y: List[int]) -> float:
        return self._model.evaluate(
            self._convert_text(x), 
            to_categorical(y, len(self._classes)) if len(self._classes) > 2 else y)[1]

class DlConvTextClassifier(DlTensorTextClassifier):
    def _create_tokenizer(self):
        return Tokenizer(
            num_words=self.vocab_size, 
            oov_token=self.oov_tok)

    def create_model(self):
        layers = [
            tf.keras.layers.Embedding(
                self.vocab_size, 
                self.embedding_dim, 
                input_length=self.max_length),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
        ]
        if len(self._classes) > 2:
            layers.append(tf.keras.layers.Dense(len(self._classes), activation='softmax'))
        else:
            layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

        return tf.keras.Sequential(layers)

    def _convert_text(self, x) -> List[List[float]]:
        return pad_sequences(
            self.tokenizer.texts_to_sequences(x),
            maxlen=self.max_length, 
            padding=self.padding_type, 
            truncating=self.trunc_type)