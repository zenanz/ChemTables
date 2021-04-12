import torch
from ResNet import TBResNet18
from tqdm import tqdm
from torch.utils.data import DataLoader
from Sampler import BalancedBatchSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

class DWACTrainer:
    def __init__(self, model, datasets, args):
        self.model = model
        self.datasets = datasets
        self.data_loaders = dict()
        mode_list = ['calibration', 'validation', 'test']
        for idx, mode in enumerate(mode_list):
            self.data_loaders[mode] = DataLoader(self.datasets[idx],
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=4
                                                )
        class_vector = self.datasets[0].class_vector
        class_balanced_sampler = BalancedBatchSampler(self.datasets[0], class_vector)
        self.data_loaders['train'] = DataLoader(self.datasets[idx],
                                            batch_size=args.batch_size,
                                            sampler=class_balanced_sampler,
                                            num_workers=4
                                            )
        self.n_classes = args.n_classes
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.patience = args.patience
        self.optim = torch.optim.Adam((x for x in self.model.parameters() if x.requires_grad), args.lr)
        self.target_names = self.model.embedder._indexer._target_names

    def train(self, epoch_idx):
        self.model.train()
        pbar = tqdm(iter(self.data_loaders['train']))
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(pbar):
            self.optim.zero_grad()
            data = tuple(t.to(self.device) for t in data)
            target = torch.flatten(target.to(self.device))
            output_dict = self.model(data, target)
            loss = output_dict['loss']
            loss.backward()
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            self.optim.step()
            pbar.set_description('::Train Epoch {}: Loss {:.4f}::'.format(epoch_idx, avg_loss))
        return total_loss

    def evaluate(self, eval_set):
        self.model.eval()
        print('\n:: %s ::' % eval_set)
        train_loader = tqdm(iter(self.data_loaders['calibration']), desc='Embedding Training Set')
        test_loader = tqdm(iter(self.data_loaders[eval_set]), desc='Embedding Evaluation Set')

        with torch.no_grad():
            ref_zs = [(self.model.get_representation(tuple(t.to(self.device) for t in x)).cpu(), y)
                                            for x, y in train_loader]

            test_zs, test_ys = zip(*[(self.model.get_representation(tuple(t.to(self.device) for t in x)).cpu(), y)
                                            for x, y in test_loader])
            class_dists = []

            for z in tqdm(test_zs, desc=':: Calculating Class Distribution ::'):
                z = z.to(self.device)

                batch_class_dists = torch.zeros([z.shape[0], self.n_classes], device=z.device)
                for ref_z, ref_y in ref_zs:
                    ref_z, ref_y = ref_z.to(self.device), ref_y.to(self.device)
                    batch_output = self.model.classify(z, ref_z, ref_y)
                    batch_class_dists.add_(batch_output['class_dists'])
                class_dists.append(batch_class_dists.cpu())

            test_zs = torch.cat(test_zs, dim=0)
            test_ys = torch.cat(test_ys, dim=0)
            test_ys = torch.flatten(test_ys).detach().to('cpu').numpy()

            class_dists = torch.cat(class_dists, dim=0)
            probs = class_dists.div(class_dists.sum(dim=1, keepdim=True)).log()
            predictions = probs.argmax(dim=1).detach().to('cpu').numpy()

            total_loss = self.model.criterion(probs, test_ys)
            report = classification_report(test_ys.tolist(), predictions.tolist(), target_names=self.target_names, digits=4)
            print(report)
            output_dict = {
                    # 'zs': test_zs,
                    # 'ys': test_ys,
                    # 'probs': probs,
                    # 'predictions': predictions,
                    # 'confs': class_dists,
                    'total_loss': total_loss,
                    'loss': total_loss.div(test_ys.shape[0]),
                    # 'correct': correct,
                    # 'accuracy': 100 * correct / test_ys.shape[0],
                    }
            # print('::Accuracy: {:.4f}::'.format(output_dict['accuracy']))
        return output_dict

    # operations each epoch on one dataloader
    def fit(self):

        no_improve = 0
        best_epoch = 0
        best_val_f1 = 0
        best_test_res = None

        for epoch_idx in range(1, self.num_epochs+1):
            if no_improve == self.patience:
                break

            res = dict()
            # train, validation and test epoch
            self.train(epoch_idx)
            validation_output = self.evaluate('validation')
            test_output = self.evaluate('test')

            if validation_output['accuracy'] > best_val_f1 or epoch_idx == 1:
                best_val_f1 = validation_output['accuracy']
                best_test_res = test_output
                best_epoch = epoch_idx
                # torch.save(self._model, os.path.join(self._serialization_dir, 'checkpoint'))
                no_improve = 0
            else:
                no_improve += 1

            # total_res.append(res)
            # self._scheduler.step()
        return (best_epoch, best_val_f1, best_test_res)



class DWACWrapper(torch.nn.Module):
    def __init__(self, embedder, model, args):
        super(DWACWrapper, self).__init__()
        self.device = args.device
        self.eps = args.eps
        self.gamma = args.gamma
        self.z_dim = args.z_dim
        self.n_classes = args.n_classes

        self.embedder = embedder
        self.model = model
        self.projection = torch.nn.Linear(model._output_dim, self.z_dim)

        self.distance_metric = self._gaussian_kernel
        self.criterion = torch.nn.NLLLoss(reduction='sum')


    def get_representation(self, x):
        x = self.embedder(x)
        x = self.model(x)
        z = self.projection(x)
        return z

    def forward(self, x, y):
        z = self.get_representation(x)

        norm = z.pow(2).sum(dim=1)
        fast_dists = torch.mm(z, z.t()).mul(-2).add(norm).t().add(norm)
        fast_dists = self.distance_metric(fast_dists)
        fast_dists = fast_dists.mul((1 != torch.eye(z.shape[0], device=z.device)).float())

        class_mask = torch.zeros(z.shape[0],
                                 self.n_classes,
                                 device=z.device)
        class_mask.scatter_(1, y.view(z.shape[0], 1), 1)
        class_dists = torch.mm(fast_dists, class_mask).add(self.eps)  # [batch_size, n_classes]
        probs = torch.div(class_dists.t(), class_dists.sum(dim=1)).log().t()
        total_loss = self.criterion(probs, y)

        output_dict = {
                'probs': probs,
                'loss': total_loss.div(y.shape[0]),
                'total_loss': total_loss,
                }

        return output_dict

    def classify(self, z, ref_z, ref_y):
        z_norm = z.pow(2).sum(dim=1)
        ref_norm = ref_z.pow(2).sum(dim=1)
        dists = torch.mm(z, ref_z.t()).mul(-2).add(ref_norm).t().add(z_norm).t()
        dists = self.distance_metric(dists)

        class_mask = torch.zeros(ref_z.shape[0],
                                 self.n_classes,
                                 device=ref_z.device)
        class_mask.scatter_(1, ref_y.view(ref_z.shape[0], 1), 1)
        class_dists = torch.mm(dists, class_mask)

        output_dict = {
                'class_dists': class_dists,
                }
        return output_dict

    def _gaussian_kernel(self, dists):
        return dists.mul_(-1 * self.gamma).exp_()
