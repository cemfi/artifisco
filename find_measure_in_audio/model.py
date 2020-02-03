import json
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet101, resnet50, inception_v3, resnet34

from dataset import ArtifiscoDataset
import statistics


class AudioMeasureFinderModel(pl.LightningModule):
    def __init__(self, root, batch_size):
        super(AudioMeasureFinderModel, self).__init__()
        self.batch_size = batch_size
        self.root = root

        self.window_count = 646

        # with open(os.path.join(root, 'metadata.json')) as fp:
        #     self.metadata = json.load(fp)

        # self.window_count = self.metadata['window_count_spectrums']

        self.base_image = list(resnet50(pretrained=False).children())[:-1]
        self.base_image = nn.Sequential(*self.base_image)
        # self.base_image = inception_v3(pretrained=False, num_classes=self.window_count, aux_logits=False, transform_input=True)

        self.base_spectrum = list(resnet50(pretrained=False).children())[:-1]
        self.base_spectrum = nn.Sequential(*self.base_spectrum)
        # self.base_spectrum = inception_v3(pretrained=False, num_classes=self.window_count, aux_logits=False, transform_input=True)

        self.fc_1 = nn.Linear(4096, 512)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.fc_end = nn.Linear(512, self.window_count + 1)

        self.loss_end = nn.CrossEntropyLoss()

    def forward_image(self, x):
        y = self.base_image(x)
        y = torch.flatten(y, 1)
        return y

    def forward_spectrum(self, x):
        y = self.base_spectrum(x)
        y = torch.flatten(y, 1)
        return y

    def forward(self, x):
        y_image = self.forward_image(x['image'])
        y_spectrum = self.forward_spectrum(x['spectrum'])

        y = torch.cat([y_image, y_spectrum], dim=1)
        y = self.fc_1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc_end(y)
        return y

    def step(self, batch, batch_idx):
        y_hat_end = self.forward(batch)
        y_end = batch['target']
        loss_end = self.loss_end(y_hat_end, y_end)
        losses = loss_end

        accuracy = torch.abs(y_hat_end.argmax(dim=1) - y_end.float()).mean().item()

        accuracy = 1 - (accuracy / self.window_count)

        return losses, accuracy

    def training_step(self, batch, batch_idx):
        losses, accuracy = self.step(batch, batch_idx)
        tensorboard_logs = {
            'train_loss': losses,
            'train_accuracy': accuracy
        }
        return {'loss': losses, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        losses, accuracy = self.step(batch, batch_idx)
        return {'val_loss': losses, 'val_acc': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = statistics.mean([x['val_acc'] for x in outputs])
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_acc': avg_acc
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    @pl.data_loader
    def train_dataloader(self):
        dataset= ArtifiscoDataset(
                root=self.root,
                train_val_split=0.9,
                train=True
            )
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=self.batch_size,
        )

    @pl.data_loader
    def val_dataloader(self):
        dataset = ArtifiscoDataset(
            root=self.root,
            train_val_split=0.9,
            train=False
        )
        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        return DataLoader(
            dataset,
            sampler=dist_sampler,
            batch_size=self.batch_size,
        )
