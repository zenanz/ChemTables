import argparse
import os
import sys
import random
import pickle
import torch
import json
import numpy as np
from TBDataset import TBDataset
from PretrainedWordVectors import PretrainedWordVectors
from Embedding import CellEmbedder, TRMCellEmbedder
from Indexer import TableWordIndexer
from ResNet import TBResNet18
from DWAC import DWACTrainer, DWACWrapper


parser = argparse.ArgumentParser(description='ChemTables Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch_size', type=int, default=15, metavar='N',
                        help='training batch size')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                        help='learning rate for training')
parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                        help='number of training epochs')
parser.add_argument('--height_limit', type=int, default=10, metavar='N',
                        help='maximum height of tables')
parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='number of training epochs')

parser.add_argument('--seq_length_limit', type=int, default=20, metavar='N',
                    help='maximun number of tokens in each cell')
parser.add_argument('--z_dim', type=int, default=5, metavar='N',
                    help='dimensions of latent representation')
parser.add_argument('--gamma', type=float, default=1, metavar='k',
                    help='hyperparameter for kernel')
parser.add_argument('--eps', type=float, default=1e-12, metavar='k',
                    help='label smoothing factor for learning')
parser.add_argument('--device', type=int, default=0, metavar='N',
                    help='GPU ID')
parser.add_argument('--n_classes', type=int, default=5, metavar='N',
                    help='Number of classes in the dataset')
parser.add_argument('--learning_curve', type=bool, defualt=False, metavar='B',
                    help="test learning curve # of folds used for training")

args = parser.parse_args()

# Reproducibility
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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

model_dir = os.path.join('model', 'dwacresnet')

fold_idx = 0

if os.path.exists(dataset_cache) and os.path.exists(pre_trained_cache):
    dataset = pickle.load(open(dataset_cache, 'rb'))
    print('::Loaded Dataset::')
    pre_trained_vector = pickle.load(open(pre_trained_cache, 'rb'))
    print('::Loaded Pre-trained Vectors::')

else:
    dataset = TBDataset(data_dir)
    pre_trained_vector = PretrainedWordVectors(dataset, pretrained_path=w2v_dir)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    pickle.dump(dataset, open(dataset_cache, 'wb'))
    pickle.dump(pre_trained_vector, open(pre_trained_cache, 'wb'))

indexer = TableWordIndexer(dataset,
                        seq_length_limit = args.seq_length_limit,
                        table_size_limit = (args.height_limit,32),
                        use_pretrained=pre_trained_vector)

num_classes = indexer._num_classes
num_folds = len(dataset._fold_indices)
dataset.set_indexer(indexer)

if args.learning_curve = False:
    datasets = dataset.train_test_split(fold_idx)
else:
    datasets = dataset.train_test_split_curve(fold_idx)


embedder = CellEmbedder(indexer)
model = DWACWrapper(embedder, TBResNet18(embedder._output_dim), args)
model.cuda()

print(model)
create_dir('model')
create_dir(model_dir)
print(model_dir)

trainer = DWACTrainer(model, datasets, args)

best_test = trainer.fit()

# print('::Best Epoch %d::' % best_test[0])
# print('::Best Test F1 %f::' % best_test[2]['f1'])
# print('::Best Test Classification Report::')
# print(best_test[2]['report'])

# res_path = os.path.join(model_dir, 'result_list.json')
best_path = os.path.join(model_dir, 'best_results.json')
# res_file = open(res_path, 'w+')
best_file = open(best_path, 'w+')
# res_file.write(json.dumps(result))
best_file.write(json.dumps(best_test))
# res_file.close()
best_file.close()
