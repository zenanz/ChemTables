import logging
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig

class TableBert(torch.nn.Module):
    def __init__(self, pre_trained_model_name, cache_dir='cache/'):
        super(TableBert, self).__init__()
        self.config = BertConfig.from_pretrained(pre_trained_model_name)
        self._tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name, do_lower_case=False, cache_dir=cache_dir)
        self._model = BertForSequenceClassification.from_pretrained(pre_trained_model_name, config=self.config, cache_dir=cache_dir)
        # surpress warning for when tokenizing long sequences
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

        # dropout_prob = self._model.config.hidden_dropout_prob
        # hidden_size = self._model.config.hidden_size
        # self._projection_layer = torch.nn.Linear(hidden_size, hidden_size)
        # self._dp = torch.nn.Dropout(dropout_prob)

    def forward(self, inputs):
        x = self._model(**inputs)[0] # pooler outputs
        # x = self._dp(x)
        # x = self._projection_layer(x)

        return x
