from tqdm import tqdm
import os
import sys
import pickle
import logging
import numpy as np
import random
import torch
import json
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup
# from pytorch_transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

max_sequence_length = sys.argv[1]
mode = sys.argv[2]

# fixed random seeds for reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

# load datasets
label2idx = pickle.load(open('cache/label2idx.pkl', 'rb'))
idx2label = {label2idx[label]:label for label in label2idx} # create reverse mapping from label index to labels in training set
target_names = [idx2label[i] for i in range(len(idx2label))]
train_set = pickle.load(open('cache/%s_train.pkl' % mode, 'rb'))
dev_set = pickle.load(open('cache/%s_dev.pkl' % mode, 'rb'))
test_set = pickle.load(open('cache/%s_test.pkl' % mode, 'rb'))
print("::Loaded datasets::")

# load pretrained transformer
model_name = 'bert-base-multilingual-cased'
config = BertConfig.from_pretrained(model_name, num_labels=len(label2idx))
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertForSequenceClassification.from_pretrained(model_name, config=config, cache_dir='cache').cuda()
print("::Loaded BERT from pre-trained file::")

# load pretrained transformer
# config = XLNetConfig.from_pretrained('xlnet-large-cased', num_labels=len(label_map))
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=False)
# model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased', config=config, cache_dir='cache').cuda()
# print("::Loaded XLNet from pre-trained file::")

print(model)

# Multi GPU Training
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    print("::Multi-GPU Training on %d devices::" % n_gpu)

patience = 5
num_train_epochs = 50
fold = 3

train_batch_size = 2*n_gpu
serialization_dir = 'model/%s_%s' % (mode, max_sequence_length)

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(serialization_dir):
    os.mkdir(serialization_dir)

train_dataloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
eval_dataloader = DataLoader(dev_set, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=train_batch_size, shuffle=True)

# Set weight decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*num_train_epochs)
best_result = 0.0

# training
# for epoch_idx in range(num_train_epochs):

def epoch(epoch_idx, dataloader, mode):
    total_loss = 0.0
    mean_loss = 0.0

    label_list = []
    pred_list  = []
    table_ids  = []

    epoch_iterator = tqdm(dataloader)
    for step, batch in enumerate(epoch_iterator):
        model.zero_grad()
        if mode == 'train':
            model.train()
        else:
            model.eval()

        batch = tuple(t.to(device) for t in batch)
        table_id = batch[0].detach().cpu().numpy().tolist()
        inputs = {
                  'input_ids':      batch[1],
                  'attention_mask': batch[2],
                  'token_type_ids': batch[3],
                  'labels':         batch[4]
                 }
        outputs = model(**inputs)
        loss, logits = outputs[:2]

        preds  = np.argmax(logits.detach().cpu().numpy(), axis=1)
        labels = inputs['labels'].detach().cpu().numpy()

        pred_list  += preds.tolist()
        label_list += labels.tolist()
        table_ids += table_id

        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training

        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        mean_loss = loss.item()

        f1 = f1_score(label_list, pred_list, average='micro')
        epoch_iterator.set_description('::{} Epoch {}: Loss {:.4f} F1 {:.4f}::'.format(mode, epoch_idx, mean_loss, f1))

    output_dict = {
        'predictions': pred_list,
        'labels': label_list,
        'table_ids': table_ids,
    }

    print('::{} Summary for Epoch {}::'.format(mode, epoch_idx))
    report = classification_report(label_list, pred_list, target_names=target_names, digits=4)
    confusion = confusion_matrix(label_list, pred_list)
    print(report)

    return f1, report, confusion.tolist(), mean_loss, output_dict

def train():
    data_loaders = {
                   'train': train_dataloader,
                   'validation': eval_dataloader,
                   'test': test_dataloader
                   }

    total_res = []
    no_improve = 0
    best_epoch = 0
    best_val_f1 = 0
    best_test_res = None

    for epoch_idx in range(1, num_train_epochs+1):
        if no_improve == patience:
            break

        res = dict()
        # train, validation and test epoch
        for mode, loader in data_loaders.items():
            res[mode] = dict()
            res[mode]['f1'], res[mode]['report'], res[mode]['confusion'], res[mode]['avg_loss'], res[mode]['output_dict'] = epoch(epoch_idx, loader, mode=mode)

        if res['validation']['f1'] > best_val_f1 or epoch_idx == 1:
            best_val_f1 = res['validation']['f1']
            best_test_res = res['test']
            best_epoch = epoch_idx
            no_improve = 0
            #
            model.save_pretrained(serialization_dir)
            tokenizer.save_pretrained(serialization_dir)
        else:
            no_improve += 1

        total_res.append(res)

    return total_res, (best_epoch, best_val_f1, best_test_res)

result, best_test = train()
print('::Best Epoch %d::' % best_test[0])
print('::Best Test F1 %f::' % best_test[2]['f1'])
print('::Best Test Classification Report::')
print(best_test[2]['report'])

res_path = os.path.join(serialization_dir, 'result_list.json')
best_path = os.path.join(serialization_dir, 'best_results.json')
res_file = open(res_path, 'w+')
best_file = open(best_path, 'w+')
res_file.write(json.dumps(result))
best_file.write(json.dumps(best_test))
res_file.close()
best_file.close()
