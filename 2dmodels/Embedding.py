import torch
import torch.nn.functional as F

class WordEmbedding(torch.nn.Module):
    def __init__(self, indexer):
        super(WordEmbedding, self).__init__()
        self._indexer = indexer
        self._pre_trained_vector = self._indexer._pretrained
        self._wv_dim = self._pre_trained_vector._dim
        fixed_wv_weights = torch.stack(self._pre_trained_vector._weights)
        self._num_wv_fixed = fixed_wv_weights.size(0)
        self._num_wv_trainable = len(self._pre_trained_vector._addtional_tok)
        wv_weight = torch.empty(self._num_wv_fixed + self._num_wv_trainable, self._wv_dim)
        wv_weight[:self._num_wv_fixed] = fixed_wv_weights
        self._trainable_wv_weight= torch.nn.Parameter(torch.empty(self._num_wv_trainable, self._wv_dim))
        torch.nn.init.xavier_uniform_(self._trainable_wv_weight)
        wv_weight[self._num_wv_fixed:] = self._trainable_wv_weight
        self.register_buffer('wv_weight', wv_weight)
        self.register_parameter('trainable_wv_weight', self._trainable_wv_weight)

    def forward(self, inputs):
        wv_weight = self.wv_weight.detach()
        wv_weight[self._num_wv_fixed:] = self._trainable_wv_weight
        we = torch.nn.functional.embedding(inputs, wv_weight, padding_idx=0) # (B, H, W, S, E)
        return we


class CharacterEmbedding(torch.nn.Module):
    def __init__(self, indexer, embedding_dim=30, output_dim=50, ngram_filter_sizes = (3,)):
        super(CharacterEmbedding, self).__init__()
        self._indexer = indexer
        self._char_emb_dim = 30
        self._cv_dim = 50
        self._ngram_filter_sizes = ngram_filter_sizes
        self._num_filters = int(self._cv_dim / len(self._ngram_filter_sizes))
        self._char_vocab_size = len(self._indexer._char2idx)
        self._char_embeddings = torch.nn.Embedding(self._char_vocab_size,
                                                   self._char_emb_dim,
                                                   padding_idx=0)
        self._convolution_layers = [torch.nn.Conv1d(in_channels=self._char_emb_dim,
                                                    out_channels=self._num_filters,
                                                    kernel_size=ngram_size,
                                                    bias=False)
                                                    for ngram_size in self._ngram_filter_sizes]

        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

    def forward(self, inputs):
        batch, height, width, sequence_length, char_length = inputs.size()

        ce = self._char_embeddings(inputs) # (B, H, W, S, C, E_c)
        ce = ce.reshape(batch*height*width*sequence_length, char_length, self._char_emb_dim) # (B*H*W*S, C, E_c)

        conv_inputs = torch.transpose(ce, 1, 2) # (B*H*W*S, E_c, C)
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(
                    torch.nn.functional.relu(convolution_layer(conv_inputs)).max(dim=2)[0]
            )

        ce = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0] # (B*H*W*S, E)
        ce = ce.reshape(batch, height, width, sequence_length, -1) # (B, H, W, S, E)
        return ce

class WordAttention(torch.nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super(WordAttention, self).__init__()
        self._projection = torch.nn.Linear(hidden_dim*2, attention_dim, bias=True)
        self._uv = torch.nn.Parameter(torch.empty(attention_dim, 1))
        torch.nn.init.xavier_uniform_(self._uv)
        self.register_parameter('context_vector', self._uv)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        ut = self._projection(inputs)
        att = torch.bmm(ut, self._uv.expand(batch_size, -1, -1))
        att_weights = F.softmax(att, dim=1)
        x = inputs * att_weights
        x = torch.sum(x, dim=1)
        return x

class CellEmbedder(torch.nn.Module):
    def __init__(self, indexer, hidden_size=64, num_layers=1, char_emb=True, attention_dim=None):
        # pre-trained word vectors
        super(CellEmbedder, self).__init__()
        self._indexer = indexer
        if indexer._pretrained:
            self._word_embedder = WordEmbedding(indexer)
            self._wv_dim = self._word_embedder._wv_dim
        else:
            self._word_embedder = torch.nn.Embedding(len(indexer._word2idx), 200, padding_idx=0)
            self._wv_dim = 200

        # Character level word embedder
        self._char_embedder = CharacterEmbedding(indexer) if char_emb else None
        self._cv_dim = self._char_embedder._cv_dim if self._char_embedder is not None else 0

        # LSTM encoder
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.cell_bilstm = torch.nn.LSTM(input_size=self._wv_dim + self._cv_dim,
                                          hidden_size=self._hidden_size,
                                          num_layers=self._num_layers,
                                          batch_first=True,
                                          bidirectional=True)

        # Attention Layer
        self._attention_dim = attention_dim if attention_dim != -1 else self._hidden_size * 2
        self._word_attention = WordAttention(hidden_size, self._attention_dim) if attention_dim is not None else None

        # Positional embedding
        # self._positional_embeddings = torch.nn.Parameter(torch.empty(32, self._indexer._h_limit, self._indexer._w_limit))
        # torch.nn.init.xavier_uniform_(self._positional_embeddings)

        # Output dimension
        self._output_dim = hidden_size * 2 if self._word_attention is None else self._attention_dim
        # self._output_dim += 32

    def forward(self, inputs):
        self.cell_bilstm.flatten_parameters()
        word_input, char_input = inputs
        batch, height, width, sequence_length, char_length = char_input.size()
        we = self._word_embedder(word_input) # (B, H, W, S, E)
        x = we
        # If character embedding in use
        if self._char_embedder is not None:
            ce = self._char_embedder(char_input) # (B, H, W, S, E)
            x = torch.cat((we, ce), dim=4)

        x = x.reshape(batch*height*width, sequence_length, self._wv_dim+self._cv_dim)     # (B*H*W, S, E)
        output, (h_n, c_n) = self.cell_bilstm(x)

        if self._word_attention is None:
            x = h_n.transpose(0, 1)                                              #（B*H*W, 2, h)
            x = x.reshape(batch, height, width, 2*self._hidden_size)             #（B, H, W, 2xh)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self._word_attention(output)
            x = x.reshape(batch, height, width, self._attention_dim) #（B, H, W, A)
            x = x.permute(0, 3, 1, 2)
        # pe = self._positional_embeddings.expand(batch, -1, -1, -1)
        # x = torch.cat((x, pe), dim=1)

        return x

class TRMCellEmbedder(torch.nn.Module):
    def __init__(self, indexer, hidden_size=64, num_layers=1, char_emb=True, attention_dim=None, bert_embedder_dim=None, table_bert_model=None):
        # pre-trained word vectors
        super(TRMCellEmbedder, self).__init__()
        self._cell_embedder = CellEmbedder(indexer, hidden_size=64, num_layers=1, char_emb=True, attention_dim=None)
        self._indexer = indexer
        self._bert_embedder = bert_embedder_dim
        self._bert_embedder_projection = torch.nn.Linear(bert_embedder_dim, self._cell_embedder._output_dim) if bert_embedder_dim is not None else None
        # self._output_dim = 2 * self._cell_embedder._output_dim if bert_embedder_dim is not None else self._cell_embedder._output_dim
        self._output_dim = self._cell_embedder._output_dim
        self._table_bert = table_bert_model

    def forward(self, inputs):
        if self._table_bert is not None and self._bert_embedder is not None:
            word_inputs, char_input, table_bert_input, trm_outs = inputs
            table_bert_output = self._table_bert(table_bert_input)
            cell_outs = self._cell_embedder((word_inputs, char_input))
            trm_outs = self._bert_embedder_projection(trm_outs)
            trm_outs = trm_outs.permute(0, 3, 1, 2) # permute to channel first to match input shape of conv layers
            x = torch.cat((cell_outs, trm_outs), dim=1)
            return x, table_bert_output

        elif self._table_bert is not None and self._bert_embedder is None: # Only table-bert in use
            word_inputs, char_input = inputs[:2]
            table_bert_input = inputs[2:]
            cell_outs = self._cell_embedder((word_inputs, char_input))
            table_bert_input = {'input_ids':      table_bert_input[0],
                                'attention_mask': table_bert_input[1],
                                'token_type_ids': table_bert_input[2],
                                # 'labels':         table_bert_input[3]
                                }
            table_bert_output = self._table_bert(table_bert_input)
            return cell_outs, table_bert_output

        elif self._table_bert is None and self._bert_embedder is not None: # only bert embedder in use
            word_inputs, char_input, trm_outs = inputs
            # cell_outs = self._cell_embedder((word_inputs, char_input))
            trm_outs = self._bert_embedder_projection(trm_outs)
            trm_outs = trm_outs.permute(0, 3, 1, 2) # permute to channel first to match input shape of conv layers
            # x = torch.cat((cell_outs, trm_outs), dim=1)
            return trm_outs
