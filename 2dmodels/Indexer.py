import os
import torch
import pickle
import numpy as np
import itertools
from collections import Counter
from tqdm import tqdm
from TBDataset import TBDataset
from PretrainedWordVectors import PretrainedWordVectors

class TableWordIndexer(object):
    def __init__(self,
                dataset,
                use_pretrained = None,
                seq_length_limit = 50,
                word_char_limit = 25,
                table_size_limit = (32,32),
                ):

        self._dataset = dataset
        self._pretrained = use_pretrained
        self._seq_length_limit = seq_length_limit
        self._word_char_limit = word_char_limit
        self._h_limit = table_size_limit[0]
        self._w_limit = table_size_limit[1]
        self._word2idx = {}
        self._word2charindices = {}
        self._category2idx = {category:idx for idx, category in enumerate(self._dataset._label_frequency)}
        # for classification_report
        self._idx2category = {idx:category for category, idx in self._category2idx.items()}
        self._num_classes = len(self._category2idx)
        self._target_names = [self._idx2category[i] for i in range(self._num_classes)]
        self._trm_tokenizer = None

        if self._pretrained:
            self._word2idx = self._pretrained._word2idx
        else:
            self._word2idx['<pad>'] = 0
            self._word2idx['<empty>'] = 1
            for word in self._dataset._word_frequency:
                self._word2idx[word] = len(self._word2idx)

        # pre-compute character level indices for the Vocabulary
        self._char2idx = {'<pad>':0}
        for word in self._word2idx:
            char_indices = []
            for ch in word:
                if ch not in self._char2idx:
                    self._char2idx[ch] = len(self._char2idx)
                char_indices.append(self._char2idx[ch])
            self._word2charindices[word] = char_indices

    def index_token(self, token):
        if not self._pretrained or token in self._word2idx:
            return self._word2idx[token], self._word2charindices[token]
        elif len(token) > 25:
            return self._word2idx['<long_token>'], self._word2charindices['<long_token>']
        return self._word2idx['<unk>'], self._word2charindices['<unk>']

    def flatten_table(self, table):
        flattened = [token for row in table for cell in row for token in cell]
        return ' '.join(flattened).strip()

    # Tensor operations
    def padded_indice_tensor(self, table):
        padded_word_tensor = torch.zeros((self._h_limit, self._w_limit, self._seq_length_limit)).long()
        padded_char_tensor = torch.zeros((self._h_limit, self._w_limit, self._seq_length_limit, self._word_char_limit)).long()

        max_row_length = len(max(table, key=lambda x:len(x)))
        for h in range(min(len(table), self._h_limit)):
            row = table[h]
            row_length = len(row)
            for w in range(min(max_row_length, self._w_limit)):
                if w >= row_length or len(row[w]) == 0: # for cells that are empty
                    cell = ['<empty>'] * self._seq_length_limit
                else:
                    cell = row[w]

                for s in range(min(len(cell), self._seq_length_limit)):
                    word_index, char_indices = self.index_token(cell[s])
                    padded_word_tensor[h][w][s] = word_index # copy cell to padded indices matrix
                    # if empty, leave char seq as pads
                    if cell[s] != '<empty>':
                        padded_char_tensor[h][w][s][:len(char_indices)] = torch.LongTensor(char_indices[:self._word_char_limit])# copy cell char to padded indices matrix

        tensor_outputs = (padded_word_tensor, padded_char_tensor)
        # Table-BERT Input
        if self._trm_tokenizer is not None:
            input_ids = self._trm_tokenizer.encode(self.flatten_table(table), max_length=512, add_special_tokens=True)
            print(input_ids)
            pad_len = self._trm_tokenizer.max_len - len(input_ids)
            input_ids += [self._trm_tokenizer.pad_token_id] * pad_len
            padded_table_bert_tensor = torch.tensor(input_ids)
            tensor_outputs += (padded_table_bert_tensor, )

        return tensor_outputs

    def setTRMTokenizer(self, tokenizer):
        self._trm_tokenizer = tokenizer

    def labels2tensors(self, category):
        return torch.tensor([category, ]).long()

    def to_tensor(self, table):
        # entry (Table, category)
        tensor_indices = self.padded_indice_tensor(table['data'])

        indexed_label = self._category2idx[table['annotations']]
        indexed_label = self.labels2tensors(indexed_label)

        return (tensor_indices, indexed_label)
