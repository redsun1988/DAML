from abstract_classifiers import DlBaseClassifier
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from typing import List
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from tqdm import tqdm
import numpy as np


class DlBertClassifier(DlBaseClassifier):
    # "bert-base-multilingual-uncased"
    # "bert-base-uncased",
    # "DeepPavlov/rubert-base-cased"
    def __init__(self, bert_model:str = "bert-base-uncased") -> None:
        self._epochs = 1
        self._batch_size = 12
        self._MAX_LEN = 512
        self.bert_model = bert_model
        #'bert-base-uncased', "DeepPavlov/rubert-base-cased" "bert-base-multilingual-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(
            self.bert_model,
            do_lower_case=True)
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.device = torch.device('cuda:0')
            from torch.cuda import FloatTensor, LongTensor            # Import GPU Tensors.
        else:
            self.device = torch.device('cpu')
            from torch import FloatTensor, LongTensor                 # Import CPU Tensors.

    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, value):
        self._epochs = value
    
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def create_model(self):
        bert = BertForSequenceClassification.from_pretrained(
            self.bert_model, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 3, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        if self.is_cuda:
            bert.cuda()
        return bert 

    def _convert_text(self, x) -> List[List[float]]:
        input_ids = []
        attention_masks = []
        
        for row in x:
            encoded_row = self.tokenizer.encode(
                                row,                       # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                           )
            # Add the encoded sentence to the list.
            input_ids.append(encoded_row)

        input_ids = pad_sequences(input_ids, 
                                  maxlen=self._MAX_LEN, dtype="long", 
                                  value=0, 
                                  truncating="post", padding="post")

        
        # Make attention masks token -> 1, [PAD] -> 0
        for encoded_row in input_ids:
            att_mask = [int(token_id > 0) for token_id in encoded_row]
            attention_masks.append(att_mask)

        return input_ids, attention_masks
    
    def fit(self, x: List[str], y: List[int]) -> None:
        if not hasattr(self, "_model"):
            self._model = self.create_model()
        
        input_ids, attention_masks = self._convert_text(x)
        self.optimizer = AdamW(
            self._model.parameters(),
            lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
            )
        total_steps = len(input_ids) * self.epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 10, 
            num_training_steps = total_steps)
        
        data = TensorDataset(
            torch.tensor(input_ids), 
            torch.tensor(attention_masks), 
            torch.tensor(y))
        dataloader = DataLoader(
            data, 
            sampler=RandomSampler(data), 
            batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self._do_epoch(epoch, dataloader)

    def load_model(self, path: str):
        BertForSequenceClassification.from_pretrained(
            path, 
            num_labels = 3, # The number of output labels--2 for binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        if self.is_cuda:
            bert.cuda()
    
        self._model = bert 
            
    def score(self, prediction, labels):
        correct_samples = torch.sum(prediction == labels).cpu().numpy()
        return correct_samples / labels.shape[0]
    
    def _do_epoch(self, epoch: int, data: TensorDataset) -> float:
        accuracy = 0
        epoch_loss = 0

        batch_count = len(data)
        self._model.train(True)

        with torch.autograd.set_grad_enabled(True):
            with tqdm(total=batch_count) as progress_bar:               
                for ind, batch in enumerate(data):
                    X_batch = batch[0].to(self.device)
                    X_mask = batch[1].to(self.device)
                    y_batch = batch[2].to(self.device)
                    
                    loss, logits = self._model(X_batch, 
                                                   token_type_ids=None, 
                                                   attention_mask=X_mask, 
                                                   labels=y_batch)

                    epoch_loss += loss.item()
                    accuracy += self.score(
                        torch.argmax(logits, dim=1), 
                        y_batch)

                    self.update_weights_by_gradients(accuracy, loss)
                    
                    progress_bar.update()
                    progress_bar.set_description('Epoch {} - accuracy: {:.2f}, loss {:.2f}'.format(
                        epoch, (accuracy / (ind+1)), epoch_loss / (ind+1))
                    )

                accuracy /= (ind + 1)
                epoch_loss /= (ind + 1) 
                progress_bar.set_description(f'Epoch {epoch} - accuracy: {accuracy:.2f}, loss: {epoch_loss:.2f}')

        return accuracy
    
    def update_weights_by_gradients(self, accuracy, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(accuracy)

    def predict(self, x: List[str]) -> List[int]:
        if not self._model:
            raise RuntimeError("_model is not initialized")
        
        input_ids, attention_masks = self._convert_text(x)
        
        data = TensorDataset(
            torch.tensor(input_ids), 
            torch.tensor(attention_masks)
        )
        dataloader = DataLoader(
            data, 
            batch_size=self.batch_size)

        predictions = np.zeros(len(x))
        index = 0
        with torch.autograd.set_grad_enabled(False):
            for ind, batch in enumerate(dataloader):
                X_batch = batch[0].to(self.device)
                X_mask = batch[1].to(self.device)
                
                self._model.train(False)
                logits = self._model(X_batch, 
                                            token_type_ids=None, 
                                            attention_mask=X_mask, 
                                            )[0]
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                for label in batch_predictions:
                    predictions[index] = label
                    index += 1
        return predictions