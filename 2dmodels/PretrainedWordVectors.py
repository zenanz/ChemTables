import torch
from tqdm import trange, tqdm

class PretrainedWordVectors(object):
    def __init__(self,
                dataset,
                pretrained_path = 'patent_w2v.txt',
                oov_threshold = 5):
        self._word2idx = {'<pad>':0}
        self._oov_threshold = oov_threshold
        self._addtional_tok = ['<unk>', '<long_token>', '<empty>']

        pretrained_file = open(pretrained_path, 'r')
        self._dataset = dataset
        self._vocab_size, self._dim = map(int, pretrained_file.readline().strip().split(" "))
        self._weights = [torch.zeros(self._dim, )] # placeholder for <pad>

        # Only add vectors for words appears in dataset
        for i in trange(self._vocab_size, desc='::Loading w2v weights:'):
            line = pretrained_file.readline().strip().split(" ")
            word, weights = line[0], list(map(float, line[1:])) # extract word and weights
            if word in self._dataset._word_frequency:
                self._word2idx[word] = len(self._word2idx)
                self._weights.append(torch.FloatTensor(weights))

        # Add OOV words
        for word in tqdm(self._dataset._word_frequency, desc='::Adding OOV Words:'):
            if word not in self._word2idx and self._dataset._word_frequency[word] >= self._oov_threshold:
                self._addtional_tok.append(word)

        for token in self._addtional_tok:
            self._word2idx[token] = len(self._word2idx)
