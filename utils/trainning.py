#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from utils.data_processing import transform
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import datasets
import torch.nn.functional as F
import torch


# In[ ]:
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class dataset(Dataset):
    def __init__(self, dataset_path):
        self.images = datasets.ImageFolder(root=f'./{dataset_path}/portraits/', transform=transform) 
        self.dnas = torch.load(f'./{dataset_path}/tensors/dnas.pt')
        self.genders = torch.load(f'./{dataset_path}/tensors/genders.pt').float()
        self.n_samples = len(self.images) 

    def __getitem__(self, index):
        image, i = self.images[index]
        return image, self.genders[i], self.dnas[i]

    def __len__(self):
        return self.n_samples


# In[ ]:


class Net(LightningModule):

    def __init__(self, model):
        super(Net, self).__init__()
        self.model = model

        self.epochs = None
        self.learning_rate = None
        self.weight_decay = None

    def get_parameters(self, hyperparameters):
        print(hyperparameters)
        self.epochs = hyperparameters['epochs']
        self.learning_rate = hyperparameters['lr']
        self.weight_decay = hyperparameters['weight decay']

        

    def mse(self, x, y):
        return  F.mse_loss(x, y)
    

    def training_step(self, batch, batch_idx):
        images, genders, dnas = batch
        
        predictions = self.model(images, genders)

        loss = self.mse(predictions, dnas)       
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.logger.log_metrics({'trainning loss': avg_loss}, self.current_epoch)
        self.logger.log_metrics({'learning rate': self.scheduler._last_lr[0]}, self.current_epoch)


    def validation_step(self, batch, batch_idx):
        images, genders, dnas = batch
        
        predictions = self.model(images, genders)

        loss = self.mse(predictions, dnas)       
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('val_loss', avg_loss)
        self.logger.log_metrics({'validation loss': avg_loss}, self.current_epoch)
    
        self.scheduler.step(avg_loss)
   
    def configure_optimizers(self):
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.90, threshold_mode='abs', threshold=6)
        return {'optimizer': self.optimizer, 'scheduler': self.scheduler, 'monitor': 'val_loss'}

# In[ ]:


class datamodule(LightningDataModule):
    def __init__(self, opts):
        super(datamodule, self).__init__()
        self.val_ratio = opts.val_ratio
        self.batch_size = opts.batch_size
        self.num_workers = opts.workers
        self.dataset_path = opts.datastore_name


    def setup(self, stage=None):
        data = dataset(self.dataset_path)
        valid_idx = int(len(data)*self.val_ratio)
        train_idx = len(data)-valid_idx

        self.train_data, self.val_data = random_split(data, [train_idx, valid_idx])
        

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size, num_workers=self.num_workers)
    


# In[ ]:


class experiment():
    def __init__(self, model, opts):
        self.parameters = { 'batch size': opts.batch_size,
                            'epochs': opts.epochs,
                            'lr': opts.lr,
                            'weight decay': 0.01,
                        }


        self.model = Net(model)
        self.model.get_parameters(self.parameters)

        self.datamodule = datamodule(opts)

        self.stop = EarlyStopping(monitor='val_loss', patience=6, strict=False, mode='min', min_delta=6)
        
        self.trainer = Trainer(callbacks=[self.stop],
                                gpus=opts.gpus,
                                max_epochs=opts.epochs,
                                check_val_every_n_epoch=opts.val_epochs,
                                logger=TensorBoardLogger("./logs", name='Regressor', default_hp_metric=False, log_graph=True)
                                )
        

    def fit(self):
        self.trainer.fit(self.model, self.datamodule)


if __name__ == '__main__':
    None