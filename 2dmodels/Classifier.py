import torch
from Embedding import TRMCellEmbedder, CellEmbedder
from ResNet import TBResNet, TBResNet18
from PixelRNN import PixelRNN


class Classifier(torch.nn.Module):
    def __init__(self, embedder, model, dropout=0):
        super(Classifier, self).__init__()

        self.embedder = embedder
        self._model = model
        self._num_classes = embedder._indexer._num_classes
        # self._model_out_dim = self._num_classes
        self._model_out_dim = model[-1]._output_dim if isinstance(model, torch.nn.Sequential) else model._output_dim
        self._output_dim = self._num_classes
        self._fc = torch.nn.Linear(self._model_out_dim, self._num_classes)

    def forward(self, inputs):
        x = self.embedder(inputs)
        x = self._model(x)
        x = self._fc(x)

        return x

class HybridClassifier(Classifier):
    def __init__(self, embedder, model, dropout=0):
        super(HybridClassifier, self).__init__(embedder, model)

        # self._table_bert_dim = self.embedder._table_bert.config.hidden_size
        # self._proj = torch.nn.Linear(self._table_bert_dim, self._model_out_dim)
        #
        # self._fc = torch.nn.Sequential(
        #     torch.nn.Linear(3 * self._num_classes, self._num_classes),
        # )
        # self._x_weights = torch.nn.Parameter(torch.ones(2, self._num_classes))
        # self.softmax_d1 = torch.nn.Softmax(dim=1)
        # self.softmax_d0 = torch.nn.Softmax(dim=0)
        # torch.nn.init.xavier_uniform_(self._x_weights)
        # torch.nn.init.xavier_uniform_(self._y_weights)

    def freeze_pretrained_weights(self):
        for param in self.embedder.parameters():
            param.requires_grad_(False)

        for param in self._model.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name, param.requires_grad)

    def forward(self, inputs):
        x, y = self.embedder(inputs)
        x = self._model(inputs[:2])
        # x = self.softmax_d1(x)
        # y = self.softmax_d1(y)
        # weights = self.softmax_d0(self._x_weights)
        # y = self._proj(y) # project Table-BERT output to the same dimension as encoder output
        # print(x.size(), y.size())
        # x = self._x_weights[0].expand(x.size(0), -1) * x
        # y = self._x_weights[1].expand(y.size(0), -1) * y

        return x
