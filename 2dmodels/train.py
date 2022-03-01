import os
import sys
import random
import pickle
import torch
import json
import argparse
import numpy as np
from TBDataset import TBDataset
from PretrainedWordVectors import PretrainedWordVectors
from Embedding import CellEmbedder, TRMCellEmbedder
from Indexer import TableWordIndexer
from ResNet import TBResNet18
from PixelRNN import PixelRNN
from TabNet import TabNet
from Trainer import Trainer
from TableBert import TableBert
from Classifier import Classifier, HybridClassifier

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='tabnet or resnet')
parser.add_argument('height_limit', type=int, help='max height of the input table')
parser.add_argument('width_limit', type=int, help='max width of the input table')
parser.add_argument('--mode', type=str, default='full', help='evaluation mode of the model full, no_dev, or inference')
parser.add_argument('--weight_path', type=str, default="saved_state_dict", help='path to the saved state dict')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

# Reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

cache_dir = 'cache'
data_dir = 'data'
w2v_dir = 'patent_w2v.txt'
dataset_cache = os.path.join(cache_dir, 'dataset.pickle')
pre_trained_cache = os.path.join(cache_dir, 'pre_trained_vector.pickle')
model_type = args.model
height_limit = args.height_limit
width_limit =  args.width_limit
hidden_size = 64
fold_idx = 1
mode = args.mode
state_dict_path = args.weight_path

model_dir = os.path.join('model', "_".join((model_type, 'H_limit', str(height_limit), 'W_limit', str(width_limit), 'Fold', str(fold_idx))))

batch_size = 4*n_gpu
num_epochs = 50
dropout = 0
patience = 10 # earlystopping

seq_length_limit = 20
table_size_limit = (height_limit, width_limit)

# if os.path.exists(dataset_cache) and os.path.exists(pre_trained_cache):
#     dataset = pickle.load(open(dataset_cache, 'rb'))
#     print('::Loaded Dataset::')
#     pre_trained_vector = pickle.load(open(pre_trained_cache, 'rb'))
#     print('::Loaded Pre-trained Vectors::')
#
# else:
dataset = TBDataset(data_dir, mode=mode)
pre_trained_vector = PretrainedWordVectors(dataset, pretrained_path=w2v_dir)
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
pickle.dump(dataset, open(dataset_cache, 'wb'))
pickle.dump(pre_trained_vector, open(pre_trained_cache, 'wb'))

indexer = TableWordIndexer(dataset,
                        seq_length_limit = 20,
                        table_size_limit = table_size_limit,
                        use_pretrained=pre_trained_vector)

num_classes = indexer._num_classes
num_folds = len(dataset._fold_indices)
dataset.set_indexer(indexer)

if mode == 'full':
    datasets = dataset.train_test_split_curve(fold_idx)

elif mode == 'no_dev':
    datasets = dataset.train_test_split_no_dev()

elif mode == 'inference':
    print('inference prep')
    datasets = dataset.prep_inference()

def built_model(model_type):
    dataset.table_bert = False
    if model_type == 'qdlstm':
        embedder = CellEmbedder(indexer, char_emb=False, attention_dim=-1)
        pixelrnn = PixelRNN(embedder._output_dim,
                            table_size_limit,
                            hidden_size = 64,
                            num_lstm_layers = 1,
                            classifier=True)
        model = Classifier(embedder, pixelrnn)
    elif model_type == 'resnet':
        embedder = CellEmbedder(indexer, char_emb=False, attention_dim=-1)
        model = TBResNet18(embedder._output_dim, dropout=dropout)
        model = Classifier(embedder, model)
    elif model_type == 'tabnet':
        embedder = CellEmbedder(indexer, char_emb=False, attention_dim=-1)
        model = TabNet(embedder._output_dim, table_size_limit)
        model = Classifier(embedder, model)
    else:
        embedder = CellEmbedder(indexer, char_emb=False, attention_dim=-1)
        pixelrnn = PixelRNN(embedder._output_dim,
                            table_size_limit,
                            hidden_size = hidden_size,
                            num_lstm_layers = 1,
                            classifier=False)
        resnet   = TBResNet18(pixelrnn._output_dim,
                            crossconv=table_size_limit,
                            dropout=dropout)
        # tabnet = TabNet(pixelrnn._output_dim, table_size_limit)
        combined = torch.nn.Sequential(
                    pixelrnn,
                    resnet,
                )
        model = Classifier(embedder, combined)

    if mode == 'inference':
        model.load_state_dict(torch.load(state_dict_path))

    return model

def build_hybrid_model(model_type):
    model_type = model_type.split('_')[-1]
    table_bert = TableBert('cache/tablebert')
    # indexer.setTRMTokenizer(table_bert._tokenizer)
    print('::Loaded Pretrained Table-BERT::')

    embedder = TRMCellEmbedder(indexer, table_bert_model=table_bert)
    single_model = torch.load('cache/qdlstm+resnet_H_limit_20_W_limit_32/checkpoint')
    print('::Loaded Pretrained ResNet::')
    embedder._cell_embedder = single_model.embedder
    model = HybridClassifier(embedder, single_model, dropout=dropout)
    model.freeze_pretrained_weights()
    model.cuda()

    return model


# def built_hybrid_model(model_type):
#     model_type = model_type.split('_')[-1]
#     table_bert = TableBert('bert-base-multilingual-cased')
#     indexer.setTRMTokenizer(table_bert._tokenizer)
#
#     if model_type == 'qdlstm':
#         embedder = TRMCellEmbedder(indexer, table_bert_model=table_bert)
#         pixelrnn = PixelRNN(embedder._output_dim,
#                             table_size_limit,
#                             hidden_size = 64,
#                             num_lstm_layers = 1,
#                             classifier=False)
#         model = HybridClassifier(embedder, model)
#     elif model_type == 'resnet':
#         embedder = TRMCellEmbedder(indexer, table_bert_model=table_bert)
#         model = TBResNet18(embedder._output_dim)
#         model = HybridClassifier(embedder, model)
#     elif model_type == 'tabnet':
#         embedder = TRMCellEmbedder(indexer, char_emb=False, attention_dim=-1, table_bert_model=table_bert)
#         model = TabNet(embedder._output_dim)
#         model = HybridClassifier(embedder, model)
#     else:
#         embedder = TRMCellEmbedder(indexer, table_bert_model=table_bert)
#         pixelrnn = PixelRNN(embedder._output_dim,
#                             table_size_limit,
#                             hidden_size = 64,
#                             num_lstm_layers = 1,
#                             classifier=False)
#         resnet   = TBResNet18(pixelrnn._output_dim)
#         model = torch.nn.Sequential(
#                     pixelrnn,
#                     resnet,
#                 )
#         model = HybridClassifier(embedder, model)
#     return model

model = build_hybrid_model(model_type) if 'hybrid' in model_type else built_model(model_type)

if device != -1:
    model.cuda()

# Multi GPU Training
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    print("::Multi-GPU Training on %d devices::" % n_gpu)

print(model)
create_dir('model')
create_dir(model_dir)
print(model_dir)

if mode == 'full' or mode == 'no_dev':
    trainer = Trainer(model,
                    datasets,
                    batch_size,
                    num_epochs,
                    patience,
                    device,
                    serialization_dir = model_dir,
                    target_names = indexer._target_names,
                    learning_rate = 1e-3,
                    gamma = 0.90,
                    mode = mode,
                    n_gpu=n_gpu)
    result, best_test = trainer.train()
else:
    test_set = TBDataset('test_data', mode=mode)
    test_set.set_indexer(indexer)
    test_set = test_set.prep_inference()

    trainer = Trainer(model,
                    test_set,
                    batch_size,
                    num_epochs,
                    patience,
                    device,
                    serialization_dir = model_dir,
                    target_names = indexer._target_names,
                    learning_rate = 1e-3,
                    gamma = 0.90,
                    mode = mode,
                    n_gpu=n_gpu)
    result, best_test = trainer.predict()

print('::Best Epoch %d::' % best_test[0])
print('::Best Test F1 %f::' % best_test[2]['f1'])
print('::Best Test Classification Report::')
print(best_test[2]['report'])

res_path = os.path.join(model_dir, 'result_list.json')
best_path = os.path.join(model_dir, 'best_results.json')
res_file = open(res_path, 'w+')
best_file = open(best_path, 'w+')
res_file.write(json.dumps(result))
best_file.write(json.dumps(best_test))
res_file.close()
best_file.close()
