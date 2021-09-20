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
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

# fixed random seeds for reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

# load datasets
mode = sys.argv[2]
label2idx = pickle.load(open('cache/label2idx.pkl', 'rb'))
idx2label = {label2idx[label]:label for label in label2idx} # create reverse mapping from label index to labels in training set
target_names = [idx2label[i] for i in range(len(idx2label))]
test_set = pickle.load(open('cache/%s_test.pkl' % mode, 'rb'))
print("::Loaded datasets::")

# load pretrained transformer
model_name = sys.argv[1]
model_path = os.path.join('models', model_name)
config = BertConfig.from_pretrained(model_path, num_labels=len(label2idx))
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
model = BertForSequenceClassification.from_pretrained(model_path, config=config, cache_dir='cache').cuda()
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

num_epochs = 1

batch_size = 1*n_gpu
serialization_dir = 'models/%s' % (model_name)

if not os.path.exists('model'):
    os.mkdir('model')

if not os.path.exists(serialization_dir):
    os.mkdir(serialization_dir)

test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

best_result = 0.0

# for epoch_idx in range(num_train_epochs):

def epoch(dataloader, mode):
    total_loss = 0.0
    mean_loss = 0.0

    label_list = []
    pred_list  = []
    table_ids  = []

    epoch_iterator = tqdm(dataloader)
    for step, batch in enumerate(epoch_iterator):
        model.zero_grad()
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

        mean_loss = loss.item()

        f1 = f1_score(label_list, pred_list, average='micro')
        epoch_iterator.set_description('::{}: Loss {:.4f} F1 {:.4f}::'.format(mode, mean_loss, f1))

    output_dict = {
        'predictions': pred_list,
        'labels': label_list,
        'table_ids': table_ids,
    }

    print('::{} Summary::'.format(mode))
    report = classification_report(label_list, pred_list, target_names=target_names, digits=4)
    confusion = confusion_matrix(label_list, pred_list)
    print(report)

    return f1, report, confusion.tolist(), mean_loss, output_dict

def predict():


    best_epoch = 0
    best_val_f1 = 0
    best_test_res = None

    res = dict()
    # test epoch
    mode = 'test'
    loader = test_dataloader

    res['f1'], res['report'], res['confusion'], res['avg_loss'], res['output_dict'] = epoch(loader, mode=mode)

    return res

result = predict()
print('::Best Test F1 %f::' % result['f1'])
print('::Best Test Classification Report::')
print(result['report'])

res_path = os.path.join(serialization_dir, 'result.json')
res_file = open(res_path, 'w+')
res_file.write(json.dumps(result))
res_file.close()
