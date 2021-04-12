import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict
from CrossConv import CrossConv2d
from ResNet import TBResNet18

directionality = 4

'''
0: right down
1: left  down
2: right top
3: left  top
'''
def redirect(inputs, direction):
    if direction == 0:
        return inputs

    dims = []
    if direction // 2 == 1:
        dims.append(2)
    if direction % 2 == 1:
        dims.append(3)
    return torch.flip(inputs, dims=dims)


# DiagonalLSTMCell
class DiagonalLSTMCell(torch.nn.Module):
    def __init__(self, hidden_size, height):
        super(DiagonalLSTMCell, self).__init__()
        self._height = height
        self._hidden_size = hidden_size
        self._num_units = self._hidden_size * self._height
        self._output_size = self._num_units
        self._state2state = torch.nn.Conv1d(self._hidden_size, 4*self._hidden_size, 2)
        self._cell2cell = torch.nn.Conv1d(self._hidden_size, self._hidden_size, 2)

    def forward(self, i2s, states):
        batch_size = i2s.size(0)
        cell_state_prev = states[:, :self._num_units]   # 0:h  (batch, height*h)
        hidden_state_prev = states[:, self._num_units:] # h:2h (batch, height*h)
        # Move channel(h_size) to dim 1, matching with Conv1d
        hidden_state_prev = hidden_state_prev.reshape(-1, self._height, self._hidden_size)
        cell_state_prev = cell_state_prev.reshape(-1, self._height, self._hidden_size)
        # Pad first row for 2x1 conv (batch, 1+height, h)
        pad = torch.cuda.FloatTensor(batch_size, 1, self._hidden_size).fill_(0)
        padded_hidden_state_prev = torch.cat([pad, hidden_state_prev], dim=1)
        padded_hidden_state_prev = torch.transpose(padded_hidden_state_prev, 1, 2)

        padded_cell_state_prev = torch.cat([pad, cell_state_prev], dim=1)
        padded_cell_state_prev = torch.transpose(padded_cell_state_prev, 1, 2)

        # Upsampling by 2 conv1d
        s2s = self._state2state(padded_hidden_state_prev) # (batch, 4h, height)
        c2c = self._cell2cell(padded_cell_state_prev)
        # Move channel(h_size back to dim 2) and reshape to input size
        s2s = torch.transpose(s2s, 1, 2)
        s2s = s2s.reshape(-1, self._hidden_size*4*self._height) # (batch, height*4h)

        c2c = torch.transpose(c2c, 1, 2)
        c2c = c2c.reshape(-1, self._hidden_size*self._height) # (batch, height*h)

        lstm_matrix = torch.sigmoid(i2s + s2s)

        i, g, f, o = lstm_matrix.split(self._hidden_size*self._height, 1) # (batch, height*h) for each gate

        c = f * c2c + i * g
        h = o * torch.tanh(c)

        new_states = torch.cat([c,h], 1)
        return h, new_states


# DiagonalLSTM
class DiagonalLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, use_residual=True):
        super(DiagonalLSTM, self).__init__()
        self._input_channel, self._height, self._width = input_size
        self._new_width = self._height + self._width - 1
        self._hidden_size = hidden_size
        self._input2state = torch.nn.Conv2d(self._input_channel, 4*self._hidden_size, (1,1))
        self._cell = DiagonalLSTMCell(self._hidden_size, self._height)
        # self._residual_block = MaskedConv2d(self._hidden_size, 2*self._hidden_size, (1,1), mask_mode='B')
        self._upsampling = torch.nn.Conv2d(self._hidden_size, 2*self._hidden_size, (1,1))

    def skew(self, inputs):
        batch_size = inputs.size(0)

        rows = inputs.split(1, dim=2)
        new_rows = []

        for idx, row in enumerate(rows):
            # (batch, channel, width)
            squeezed_row = row.squeeze(2)
            # (batch*channel, width)
            reshaped_row = squeezed_row.reshape(batch_size*self._input_channel, self._width)
            # pad after and before actual row (batch*channel, new_width)
            padded_row = torch.nn.functional.pad(reshaped_row, (idx, self._height - 1 - idx))
            # (batch, channel, new_width)
            unreshaped_row = padded_row.reshape(batch_size, self._input_channel, self._width + self._height -1)
            assert unreshaped_row.size() == (batch_size, self._input_channel, self._new_width)
            new_rows.append(unreshaped_row)

        outputs = torch.stack(new_rows, dim=2) # (batch, channel, height, new_width)
        assert outputs.size() == (batch_size, self._input_channel, self._height, self._new_width)

        return outputs


    def unskew(self, inputs):
        batch_size = inputs.size(0)

        rows = inputs.split(1, dim=2)
        new_rows = []

        for idx, row in enumerate(rows):
            new_rows.append(row[:, :, :, idx:self._width+idx])

        outputs = torch.cat(new_rows, 2)
        # Note that if residual block is used, the output of RNN cells is h and (i.e. in_channel/2)
        assert outputs.size() == (batch_size, self._input_channel/2, self._height, self._width)

        return outputs

    def prepare_rnn_inputs(self, i2s):
        # (batch, new_width, height, 4h)
        column_wise_inputs = torch.transpose(i2s, 1, 3)

        batch_size, width, height, i2s_channel = column_wise_inputs.size()
        # (batch, new_width, height*4h)
        rnn_inputs = column_wise_inputs.reshape(batch_size, self._new_width, self._height*i2s_channel)
        # (batch, height*4h) * new_width
        rnn_inputs = [rnn_input.squeeze(1) for rnn_input in rnn_inputs.split(1, dim=1)]

        return rnn_inputs

    def forward(self, inputs):
        batch_size = inputs.size(0)
        # (batch, channel, height, new_width)
        skewed_inputs = self.skew(inputs)
        # (batch, 4h, height, new_width)
        i2s = self._input2state(skewed_inputs)
        # (batch, new_width, height*4h)
        rnn_inputs = self.prepare_rnn_inputs(i2s)

        rnn_outputs = []
        states = torch.cuda.FloatTensor(batch_size, 2*self._hidden_size*self._height).fill_(0)
        for rnn_input in rnn_inputs:
            # h: (batch, height*h); states: (batch, height*2h)
            h, states = self._cell(rnn_input, states)
            rnn_outputs.append(h)

        # (batch, new_width, height*h)
        rnn_outputs = torch.stack(rnn_outputs, dim=1)
        # (batch, new_width, height, h)
        rnn_outputs = rnn_outputs.reshape(batch_size, -1, self._height, self._hidden_size)
        # (batch, h, height, new_width) Permute to match input
        skewed_outputs = rnn_outputs.permute(0, 3, 2, 1)
        # (batch, h, height, width)
        outputs = self.unskew(skewed_outputs)
        outputs = self._upsampling(outputs)

        return outputs


# DiagonalQuadLSTM
class DiagonalQuadLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DiagonalQuadLSTM, self).__init__()
        self._input_channel, self._height, self._width = input_size
        self._hidden_size = hidden_size
        self._diagonalLSTM_list = torch.nn.ModuleList([DiagonalLSTM(input_size, self._hidden_size) for i in range(directionality)])
        self._directional_weights = torch.nn.Parameter(torch.empty(4, self._height, self._width))
        # self._ln = torch.nn.LayerNorm([2 * self._hidden_size, self._height, self._width])
        # self._relu = torch.nn.ReLU()
        self.register_parameter('directional_weights', self._directional_weights)
        torch.nn.init.xavier_uniform_(self._directional_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = [redirect(inputs, i) for i in range(directionality)]
        x = [lstm(x[i]) for i, lstm in enumerate(self._diagonalLSTM_list)]
        x = [redirect(x[i], i) for i in range(directionality)]
        # take weighted sum over outputs from all directions
        x = sum([x[i] * self._directional_weights[i:i+1].expand(batch_size, -1, -1, -1) for i in range(directionality)])
        # x = sum(x)
        # x = self._ln(x)
        # x = inputs + x
        # x = self._relu(x)
        return x


# Encapsulation of the entire model
class PixelRNN(torch.nn.Module):
    def __init__(self, inplane, table_size_limit, hidden_size, num_lstm_layers, dropout=0, classifier=False):
        super(PixelRNN, self).__init__()

        # Table embedder (B, seq_limit, H, W) -> (B, embedding_size, H, W)
        self._h_limit, self._w_limit = table_size_limit
        self._classifier = classifier

        self._hidden_size = hidden_size
        self._output_dim = self._hidden_size * 2 + inplane # LSTM output concatenate with embeddings
        # self._output_dim = inplane
        if self._classifier:
            self._output_dim *= 4

        DLSTM_input_size = (self._hidden_size * 2, self._h_limit, self._w_limit)

        self._dropout = torch.nn.Dropout2d(dropout)
        # Input conv 7x7 (B, embedding_size, H, W) -> (B, 2h, H, W)
        self._input_conv = CrossConv2d(inplane,
                                    self._hidden_size * 2,
                                    kernel_size = (self._h_limit * 2 - 1, self._w_limit * 2 - 1),
                                    stride=1,
                                    padding=(self._h_limit - 1 , self._w_limit - 1))
        # self._input_conv = torch.nn.Conv2d(inplane,
        #                             self._hidden_size * 2,
        #                             (7, 7), padding=(3,3))
        # self._bn0 = torch.nn.BatchNorm2d(2 * self._hidden_size)
        # self._ln0 = torch.nn.LayerNorm([inplane, self._h_limit, self._w_limit])
        # self._ln1 = torch.nn.LayerNorm([2 * self._hidden_size, self._h_limit, self._w_limit])
        # Stacked DiagonalBiLSTM (B, 2h, H, W) -> (B, 2h, H, W)
        self._stacked_LSTM = torch.nn.Sequential(OrderedDict([
            ("DiagonalQuadLSTM %d" % i, torch.nn.Sequential(
                    DiagonalQuadLSTM(DLSTM_input_size, self._hidden_size),
                    # torch.nn.BatchNorm2d(2 * self._hidden_size),
                )
            )
            for i in range(num_lstm_layers)
        ]))

    def forward(self, inputs):
        # x = self._ln0(inputs)
        x = self._input_conv(inputs)
        # x = self._ln1(x)
        x = self._stacked_LSTM(x)
        x = torch.cat((x, inputs), dim=1)

        if self._classifier:
            x = [x[:, :, 0, 0],
                 x[:, :, 0, self._w_limit-1],
                 x[:, :, self._h_limit-1, 0],
                 x[:, :, self._h_limit-1, self._w_limit-1]]
            x = torch.cat(x, dim=1)

        return x
