import os
import json
import pickle
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from collections import Counter

class TBDataset(Dataset):
    def __init__(self,
                path,
                cache_dir = 'cache/',
                table_bert = False,
                debug = False,):

        self._dataset = []
        self._trm_outputs = []
        self._trm_configs = None
        self._fold_indices = []
        self._ood_data = None

        for filename in sorted(os.listdir(path)):
            if 'fold' in filename:
                fold = json.load(open(os.path.join(path, filename), 'r'))
                self._fold_indices.append(list(range(len(self._dataset), len(self._dataset)+len(fold))))
                self._dataset += fold
            elif 'ood' in filename:
                self._ood_data = json.load(open(os.path.join(path, filename), 'r'))


        self._indexer = None
        # When debug mode is on, only use the first 10 tables
        if debug:
            self._dataset = self._dataset[:10]
        # Create Vocabulary
        self._word_frequency = Counter()
        self._label_frequency = Counter()

        for table in tqdm(self._dataset, desc='::Counting Freqs:'):
            data = table['data']
            category = table['annotations']
            # Count tokens in table
            tokens = [token for row in data for cell in row for token in cell]
            for token in tokens:
                self._word_frequency[token] += 1
            # Count category and mask in table
            self._label_frequency[category] += 1

        self.table_bert = table_bert
        if table_bert:
            train_set = pickle.load(open('cache/linear_train.pkl', 'rb'))
            dev_set = pickle.load(open('cache/linear_dev.pkl', 'rb'))
            test_set = pickle.load(open('cache/linear_test.pkl', 'rb'))

            self.table_bert_dataset = train_set + dev_set + test_set

    def get_class_vector(self):
        class_for_sampler = []
        for table in self._dataset:
            l = table['annotations']
            class_for_sampler.append(l)
        return class_for_sampler

    def load_trm_outputs(self, trm_dir):
        if os.path.exists(trm_dir):
            # load configs (hlimit, wlimit, hidden_dim of bert embeddings)
            self._trm_configs = pickle.load(open(os.path.join(trm_dir, 'config.pickle'), 'rb'))
            trm_output_dir = os.path.join(trm_dir, 'outs')
            trm_outputs = []
            for filename in tqdm(sorted(os.listdir(trm_output_dir)), desc='::loading transformer outputs:'):
                fold = pickle.load(open(os.path.join(trm_output_dir, filename), 'rb'))
                trm_outputs += fold
            self._trm_outputs = trm_outputs

    def pad_trm_outputs(self, inputs):
        padded_trm_outputs = torch.zeros(self._trm_configs['hlimit'],
                                         self._trm_configs['wlimit'],
                                         self._trm_configs['hidden_size'])
        empty_cell = torch.FloatTensor(self._trm_configs['empty_cell'])

        for h in range(len(inputs)):
            for w in range(len(inputs[h])):
                if inputs[h][w] is not None:
                    padded_trm_outputs[h][w] = torch.FloatTensor(inputs[h][w])
                # else:
                #     padded_trm_outputs[h][w] = empty_cell
        return padded_trm_outputs

    def __getitem__(self, index):
        table, label = self._indexer.to_tensor(self._dataset[index])
        # If bert embedding in use
        if len(self._trm_outputs) > 0:
            table_trm = self.pad_trm_outputs(self._trm_outputs[index])
            table += (table_trm, )

        if self.table_bert:
            table += self.table_bert_dataset[index]

        return table, label, index

    def __len__(self):
        return len(self._dataset)

    def set_indexer(self, indexer):
        self._indexer = indexer

    def train_test_split(self, fold_idx):
        num_folds = len(self._fold_indices)

        # train_indices = self._fold_indices[0] + self._fold_indices[1] + self._fold_indices[2]
        # dev_indices = self._fold_indices[3]
        # test_indices = self._fold_indices[4]
        train_folds = [(i+fold_idx) % num_folds for i in range(num_folds-2)]
        train_indices = []
        for i in train_folds:
            train_indices += self._fold_indices[i]

        dev_fold = (num_folds-2+fold_idx) % num_folds
        dev_indices = self._fold_indices[dev_fold]

        test_fold = (num_folds-1+fold_idx) % num_folds
        test_indices = self._fold_indices[test_fold]

        training_set = Subset(self, train_indices)
        class_vector = self.get_class_vector()
        training_set.class_vector = [class_vector[i] for i in train_indices]

        return (training_set, Subset(self, dev_indices), Subset(self, test_indices))

    def train_test_split_curve(self, num_train_folds):
        train_indices = []
        for i in range(num_train_folds):
            train_indices += self._fold_indices[i]
        dev_indices = self._fold_indices[3]
        test_indices = self._fold_indices[4]

        return (Subset(self, train_indices), Subset(self, dev_indices), Subset(self, test_indices))
