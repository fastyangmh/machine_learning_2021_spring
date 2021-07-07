# import
from genericpath import isfile
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from typing import Any
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from os.path import join
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


class RMSELoss(nn.Module):
    def __init__(self, eps=0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class CrossPartialResidualLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        layers = []
        for in_feat, out_feat in zip(in_features, out_features):
            layers.append(nn.Linear(in_features=in_feat,
                                    out_features=out_feat, bias=False))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat((x[:, :64], self.layers(x[:, 64:])), 1)


class Net(pl.LightningModule):
    def __init__(self, in_features, out_features, lr) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features=in_features, out_features=128, bias=False),
                                   nn.ReLU(),
                                   CrossPartialResidualLayer(
                                       in_features=[64, 32], out_features=[32, 64]),
                                   nn.Linear(in_features=128,
                                             out_features=out_features),
                                   nn.ReLU())
        self.lr = lr
        self.loss_function = RMSELoss()

    def forward(self, x) -> Any:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        self.log('test_loss', loss)


if __name__ == '__main__':
    # parameters
    random_seed = 0
    data_path = 'data/'
    dataset_name = 'ml2021spring-hw1'
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

    # read csv
    train_data = pd.read_csv(
        join(data_path, 'covid.train.csv'), dtype=np.float32)
    test_data = pd.read_csv(
        join(data_path, 'covid.test.csv'), dtype=np.float32)

    # get feature and label
    x_train, y_train = train_data.iloc[:, list(
        range(1, 41))+[58, 76]].values, train_data.iloc[:, -1].values
    x_test = test_data.iloc[:, list(range(1, 41))+[58, 76]].values
    y_train = y_train.reshape(-1, 1)

    # split train to val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size)

    # create dataset
    train_set = MyDataset(features=x_train, labels=y_train)
    val_set = MyDataset(features=x_val, labels=y_val)
    test_set = MyDataset(features=x_test, labels=None)

    # create data loader
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, pin_memory=use_cuda, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size,
                            shuffle=False, pin_memory=use_cuda, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             shuffle=False, pin_memory=use_cuda, num_workers=num_workers)

    # create model
    model = Net(in_features=x_train.shape[-1], out_features=1, lr=lr)

    # create trainer
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(monitor='val_loss', mode='min')],
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
            result.append(y_hat.cpu().data.numpy())
    result = np.concatenate(result, 0)

    # export the result to submission.csv
    result = pd.DataFrame(result, columns=['tested_positive'])
    result = result.rename_axis('id').reset_index()
    result.to_csv('submission.csv', index=False)
