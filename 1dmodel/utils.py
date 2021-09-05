import json
import random
from torch.utils.data import Subset
from tqdm import tqdm

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, table_id, input_ids, input_mask, segment_ids, label_id):
        self.table_id = table_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, input_files, mode='linear'):
        self._input_files = input_files
        self._examples = []
        self._label2idx = {}
        self._proc = self.linearization if mode == 'linear' else self.naturalization

    def linearization(self, table_data):
        tokens = [token for row in table_data for cell in row for token in cell]
        return ' '.join(tokens)

    def naturalization(self, table):
        def serialize_row(row):
            tokens = [token for cell in row for token in cell]
            return ' '.join(tokens)

        def get_column_header(table):
            caption = serialize_row(table[0])
            pointer = 1
            # find first non caption row
            while pointer < len(table):
                row = serialize_row(table[pointer])
                if row == caption:
                    pointer += 1
                else:
                    break
            # find first non empty row
            while pointer < len(table):
                row = serialize_row(table[pointer])
                if len(row.strip()) > 0:
                    return table[pointer], pointer
                pointer += 1
            return [[]], pointer

        paragraph = serialize_row(table[0]) + ". "

        if len(table) == 1:
            return paragraph
        elif len(table) == 2:
            return paragraph + serialize_row(table[1])

        elif len(table) > 2:
            c_headers, pointer = get_column_header(table)
            for r_idx in range(pointer+1, len(table)):
                row = table[r_idx]
                # if row is empty, continue
                if serialize_row(row).strip() == "":
                    continue

                r_header = 'row %d: ' % (r_idx - pointer)
                paragraph += r_header
                cells = []
                for c_idx in range(0, len(row)):
                    if row[c_idx] == []:
                        continue
                    if c_idx < len(c_headers) and c_headers[c_idx] != []:
                        c_header = " ".join(c_headers[c_idx]) + ":"
                    else:
                        c_header = 'column %d' % (c_idx + 1)
                    cell_text = " ".join(row[c_idx])
                    cells.append(" ".join((c_header, 'is', cell_text)))
                paragraph += ", ".join(cells)
                paragraph += '. '
        return paragraph.strip()

    def read_examples(self, quotechar=None):
        """Reads a tab separated value file."""
        for f in self._input_files:
            data_fold = json.load(open(f, 'r', encoding="utf-8-sig"))
            example_fold = []
            for (i,table) in tqdm(enumerate(data_fold), desc='::Reading tables from %s::' % f):
                table_data = self._proc(table['data'])
                if table['annotations'] in self._label2idx:
                    table_label = self._label2idx[table['annotations']]
                else:
                    table_label = len(self._label2idx)
                    self._label2idx[table['annotations']] = table_label
                example_fold.append(InputExample(i, table_data, label=table_label))
            self._examples.append(example_fold)
        return self._examples

    def get_label2idx(self):
        """Gets the list of labels for this data set."""
        return self._label2idx


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        table_id = example.guid * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(table_id=table_id,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features
