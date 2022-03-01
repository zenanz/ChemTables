import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
# from transformers import AdamW, WarmupLinearSchedule

class Trainer:
    def __init__(self,
                model,
                datasets,
                batch_size,
                num_epochs,
                patience,
                device,
                serialization_dir,
                mode='full',
                num_workers=4,
                target_names=None,
                learning_rate=1e-3,
                gamma=0.9,
                n_gpu=-1):

        self._num_workers = num_workers
        self._device = device
        self._datasets = datasets
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._model = model
        self._mode = mode

        self._data_loaders = dict()
        mode_list = ['train', 'validation', 'test'] if mode != 'inference' else ['test']
        for idx, mode in enumerate(mode_list):
            self._data_loaders[mode] = DataLoader(self._datasets[idx], batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

        print(self._data_loaders)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, 1, gamma=gamma)

        self._criterion = torch.nn.CrossEntropyLoss()
        self._patience = patience
        self._serialization_dir = serialization_dir
        self._weight_path = os.path.join(serialization_dir, 'saved_state_dict')
        self._target_names =  target_names
        self._ngpu = n_gpu


    # operations each epoch on one dataloader
    def epoch(self, epoch_idx, dataloader, mode='train'):
        epoch_type = mode.capitalize()
        sum_loss = 0.0
        label_list = []
        prediction_list = []
        input_ids = []
        logits_list = []

        if mode == 'train':
            self._model.train()
        else:
            self._model.eval()

        pbar = tqdm(iter(dataloader))
        for step, batch in enumerate(pbar):
            inputs, label_ids, ids = batch
            # print(raw_tables)
            inputs = tuple(t.to(self._device) for t in inputs)
            label_ids = label_ids.to(self._device)
            logits = self._model(inputs)
            logits, label_ids = logits.float(), label_ids.view(-1).long()
            loss = self._criterion(logits, label_ids)

            if self._ngpu > 1:
                loss = loss.mean()

            if mode == 'train':
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                # if self._table_bert is not None:
                #     self._bert_optimizer.step()
                #     self._bert_optimizer.zero_grad()
                #     self._bert_scheduler.step()

            sum_loss += loss.item()
            avg_loss = sum_loss / (step+1)

            predictions = logits.argmax(1).detach().cpu().tolist()
            label_ids = label_ids.detach().cpu().tolist()
            prediction_list += predictions
            label_list += label_ids
            logits_list += logits.detach().cpu().tolist()
            input_ids += ids.tolist()
            f1 = f1_score(label_list, prediction_list, average='micro')

            pbar.set_description('::{} Epoch {}: Loss {:.4f}, F1 {:.4f}::'.format(epoch_type, epoch_idx, avg_loss, f1))

        print('::{} Summary for Epoch {}::'.format(epoch_type, epoch_idx))
        report = classification_report(label_list, prediction_list, target_names=self._target_names, digits=4)
        confusion = confusion_matrix(label_list, prediction_list)
        print(report)

        output_dict = {
            'predictions': prediction_list,
            'labels': label_list,
            'logits': logits_list,
            'input_ids': input_ids,
        }

        return f1, report, confusion.tolist(), avg_loss, output_dict

    # process for train model on one fold
    def train(self):

        total_res = []
        no_improve = 0
        best_epoch = 0
        best_val_f1 = 0
        best_test_res = None
        validation_set_name = 'validation' if self._mode != 'no_dev' else 'test'

        for epoch_idx in range(1, self._num_epochs+1):
            if no_improve == self._patience:
                break

            res = dict()
            # train, validation and test epoch
            for mode, loader in self._data_loaders.items():
                if mode == 'validation' and self._mode == 'no_dev':
                    continue
                res[mode] = dict()
                res[mode]['f1'], res[mode]['report'], res[mode]['confusion'], res[mode]['avg_loss'], res[mode]['output_dict'] = self.epoch(epoch_idx, loader, mode=mode)

            if res[validation_set_name]['f1'] > best_val_f1 or epoch_idx == 1:
                best_val_f1 = res[validation_set_name]['f1']
                best_test_res = res['test']
                best_epoch = epoch_idx
                # torch.save(self._model, os.path.join(self._serialization_dir, 'checkpoint'))
                no_improve = 0
                self.save()
            else:
                no_improve += 1

            total_res.append(res)
            self._scheduler.step()
        return total_res, (best_epoch, best_val_f1, best_test_res)

    def predict(self):

        loader = self._data_loaders['test']
        res = dict()
        res['f1'], res['report'], res['confusion'], res['avg_loss'], res['output_dict'] = self.epoch(0, loader, mode='test')

        return [res], (0, res['f1'], res)



    def save(self):
        torch.save(self._model.state_dict(), self._weight_path)
