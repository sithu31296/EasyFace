import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from datasets import LFW
from models.adaface import AdaFace as AdaFaceModel
from heads.adaface import AdaFace as AdaFaceHead
from utils.metrics import Evaluator


def training_augmentation(size):
    return T.Compose([
        T.RandomCrop(size),
        T.RandomHorizontalFlip(0.5),
        T.Lambda(lambda x: x / 255),
        T.Normalize([0.5], [0.5])
    ])


def validation_augmentation(size):
    return T.Compose([
        T.CenterCrop(size),
        T.Lambda(lambda x: x / 255),
        T.Normalize([0.5], [0.5])
    ])

def train():
    pass


def evaluate():
    pass


def main():
    root = '/home/sithu/datasets/lfw-align-128'
    size = (112, 112)
    batch_size = 8
    device = torch.device('cuda')
    epochs = 10
    lr = 1e-2
    momentum = 0.9
    weight_decay = 5e-4
    step_size = 30
    gamma = 0.1

    evaluator = Evaluator()


    trainset = LFW(root, 'train', transform=training_augmentation(size))
    valset = LFW(root, 'test', transform=validation_augmentation(size))

    trainloader = DataLoader(trainset, batch_size, True, num_workers=8, drop_last=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size, num_workers=8, pin_memory=True)

    model = AdaFaceModel(size[0])
    head = AdaFaceHead(512, trainset.num_identities, m=0.4, h=0.333, s=64, t_alpha=0.01)
    model = model.to(device)
    head = head.to(device)

    loss_fn = CrossEntropyLoss()
    optimizer = SGD({"params": model.parameters(), "params": head.parameters()}, lr, momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size, gamma)

    for epoch in range(epochs):

        model.train()
        head.train()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            feats = model(images)
            preds = head(feats, labels)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # start evaluate
        model.eval()
        all_feats = []
        all_labels = []

        for images, labels in valloader:
            images = images[0].to(device)
            labels = labels[1].view(-1).numpy()

            with torch.no_grad():
                feats = model(images).cpu().numpy()
            
            all_feats.extend([f for f in feats])
            all_labels.extend([l for l in labels])

        encoded_feats = np.array(all_feats)
        encoded_labels = np.array(all_labels)

        stats = evaluator(encoded_feats, encoded_labels)
                

    