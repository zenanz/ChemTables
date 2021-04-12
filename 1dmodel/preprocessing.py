from tqdm import tqdm

import sys
import os
import json
import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
# from pytorch_transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from utils import DataProcessor, convert_examples_to_features

model_name = 'bert-base-multilingual-cased'
num_train_folds = 3
max_sequence_length = int(sys.argv[1])
mode = sys.argv[2]
config = BertConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertForSequenceClassification.from_pretrained(model_name, config=config, cache_dir='cache')

# config = XLNetConfig.from_pretrained('xlnet-large-cased')
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=False)
# model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', config=config, cache_dir='cache')

# Convert to Tensors and build dataset
def create_dataset(data_path, mode):
    input_files = [os.path.join(data_path, fname) for fname in sorted(os.listdir(data_path)) ]
    dp = DataProcessor(input_files, mode=mode)

    examples = dp.read_examples()
    label_list = dp.get_label2idx()
    print(label_list)
    pickle.dump(label_list, open('./cache/label2idx.pkl', 'wb'))

    dataset_folds = []
    for fold in tqdm(examples, desc='::Converting to TensorDatasets::'):
        features = convert_examples_to_features(fold, max_sequence_length, tokenizer, 'classification',
                                                      cls_token_at_end=False, # True for XLNET
                                                      cls_token=tokenizer.cls_token,
                                                      cls_token_segment_id=0, # 0 for others TRMs
                                                      sep_token=tokenizer.sep_token,
                                                      sep_token_extra=False, # roberta uses an extra separator
                                                      pad_on_left=False, # pad on the left for xlnet
                                                      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                      pad_token_segment_id=0, # 0 for others TRMs
                                                     )

        # features, label_map = convert_examples_to_features(examples, label_list, 128, tokenizer, 'classification',
        #                                               cls_token_at_end=True, # True for XLNET
        #                                               cls_token=tokenizer.cls_token,
        #                                               cls_token_segment_id=2, # 0 for others TRMs, 2 for xlnet
        #                                               sep_token=tokenizer.sep_token,
        #                                               sep_token_extra=False, # roberta uses an extra separator
        #                                               pad_on_left=True, # pad on the left for xlnet
        #                                               pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        #                                               pad_token_segment_id=4, # 0 for others TRMs, 4 for xlnet
        #                                              )

        all_table_ids = torch.tensor([f.table_id for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset_folds.append(TensorDataset(all_table_ids, all_input_ids, all_input_mask, all_segment_ids, all_label_ids))

    train = []
    for i in range(num_train_folds):
        train += dataset_folds[i]

    dev = dataset_folds[3]
    test = dataset_folds[4]

    pickle.dump(train, open('./cache/%s_train.pkl' % mode, 'wb'))
    pickle.dump(dev, open('./cache/%s_dev.pkl' % mode, 'wb'))
    pickle.dump(test, open('./cache/%s_test.pkl' % mode, 'wb'))

# create_dataset('../tb/data', 'linear')
create_dataset('data', mode)
