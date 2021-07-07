# import
import os
from os.path import join, isfile
import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Any
from torchmetrics import Accuracy, ConfusionMatrix
import pandas as pd
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import gc
import warnings
warnings.filterwarnings("ignore")

# class


class MyDataset(Dataset):
    def __init__(self, features, labels) -> None:
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x


class CrossPartialResidualLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features[0]
        layers = []
        for in_feat, out_feat in zip(in_features, out_features):
            layers.append(nn.Linear(in_features=in_feat,
                                    out_features=out_feat, bias=False))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat((x[:, :self.in_features], self.layers(x[:, self.in_features:])), 1)


class Net(pl.LightningModule):
    def __init__(self, in_features, out_features, lr) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=in_features, out_features=512, bias=False),
                                   nn.ReLU(),
                                   CrossPartialResidualLayer(
                                       in_features=[256, 128], out_features=[128, 256]),
                                   nn.Linear(in_features=512,
                                             out_features=out_features),
                                   nn.ReLU())
        self.lr = lr
        self.softmax = nn.Softmax(dim=-1)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.confusion_matrix = ConfusionMatrix(num_classes=out_features)

    def training_forward(self, x):
        return self.model(x)

    def forward(self, x) -> Any:
        return self.softmax(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def _parse_outputs(self, outputs, calculate_confusion_matrix):
        epoch_loss = []
        epoch_accuracy = []
        if calculate_confusion_matrix:
            y_true = []
            y_pred = []
        for step in outputs:
            epoch_loss.append(step['loss'].item())
            epoch_accuracy.append(step['accuracy'].item())
            if calculate_confusion_matrix:
                y_pred.append(step['y_hat'])
                y_true.append(step['y'])
        if calculate_confusion_matrix:
            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            confmat = pd.DataFrame(self.confusion_matrix(
                y_pred, y_true).tolist()).astype(int)
            return epoch_loss, epoch_accuracy, confmat
        else:
            return epoch_loss, epoch_accuracy

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        train_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': train_step_accuracy}

    def training_epoch_end(self, outputs):
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('training loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('training accuracy', np.mean(epoch_accuracy))

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        val_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': val_step_accuracy}

    def validation_epoch_end(self, outputs):
        epoch_loss, epoch_accuracy = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=False)
        self.log('validation loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('validation accuracy', np.mean(epoch_accuracy))

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.training_forward(x)
        loss = self.loss_function(y_hat, y)
        test_step_accuracy = self.accuracy(F.softmax(y_hat, dim=-1), y)
        return {'loss': loss, 'accuracy': test_step_accuracy, 'y_hat': F.softmax(y_hat, dim=-1), 'y': y}

    def test_epoch_end(self, outputs):
        epoch_loss, epoch_accuracy, confmat = self._parse_outputs(
            outputs=outputs, calculate_confusion_matrix=True)
        self.log('test loss', np.mean(epoch_loss))
        self.log('test accuracy', np.mean(epoch_accuracy))
        print(confmat)


if __name__ == '__main__':
    # parameters
    random_seed = 0
    data_path = 'data/'
    dataset_name = 'ml2021spring-hw2'
    val_size = 0.2
    batch_size = 32
    lr = 0.002
    use_cuda = True if torch.cuda.is_available() else False
    num_workers = 0
    gpus = -1 if use_cuda else 0
    max_epochs = 100
    save_path = 'save/'

    # download data
    if not isfile(join(data_path, '{}.zip'.format(dataset_name))):
        os.system(
            command='kaggle competitions download -c {} -p {}'.format(dataset_name, data_path))

    # unzip data
    os.system('unzip {}/{}.zip -d {}'.format(data_path, dataset_name, data_path))

    # set random seed
    pl.seed_everything(seed=random_seed)

    # load data
    x_train = np.load(
        join(data_path, 'timit_11/timit_11/train_11.npy')).astype(np.float32)
    y_train = np.load(
        join(data_path, 'timit_11/timit_11/train_label_11.npy')).astype(np.int)
    x_test = np.load(
        join(data_path, 'timit_11/timit_11/test_11.npy')).astype(np.float32)
    in_features = x_train.shape[-1]
    out_features = len(np.unique(y_train))

    # split train to val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size)

    # create dataset
    train_set = MyDataset(features=x_train, labels=y_train)
    val_set = MyDataset(features=x_val, labels=y_val)
    test_set = MyDataset(features=x_test, labels=None)

    # delete useless variable
    del x_train, y_train, x_val, y_val, x_test
    gc.collect()

    # create data loader
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, pin_memory=use_cuda, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size,
                            shuffle=False, pin_memory=use_cuda, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             shuffle=False, pin_memory=use_cuda, num_workers=num_workers)

    # create model
    model = Net(in_features=in_features, out_features=out_features, lr=lr)

    # create trainer
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor='validation loss', mode='min')],
                         gpus=gpus,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=1,
                         default_root_dir=save_path)

    # train model
    trainer.fit(model, train_loader, val_loader)

    # test model
    for stage, data_loader in zip(['train data', 'val data'], [train_loader, val_loader]):
        print('the {} test result is:'.format(stage))
        trainer.test(test_dataloaders=data_loader)
        print('\n')

    # predict the test_set
    model.eval()
    result = []
    for x in test_loader:
        with torch.no_grad():
            y_hat = model(x)
            y_hat = torch.argmax(y_hat, dim=-1)
            result.append(y_hat.cpu().data.numpy())
    result = np.concatenate(result, 0)

    # export the result to submission.csv
    result = pd.DataFrame(result, columns=['Class'])
    result = result.rename_axis('Id').reset_index()
    result.to_csv('submission.csv', index=False)
