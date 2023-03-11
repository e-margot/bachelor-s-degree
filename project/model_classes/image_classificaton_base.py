import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from operator import truediv
from loguru import logger
import config


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss
    
    def __init__(self):
        super().__init__()
        nb_classes = config.NUM_CLS
        self.matrix = torch.zeros(nb_classes, nb_classes)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        temp_labels = labels.softmax(dim=1)
        loss = F.cross_entropy(out, temp_labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        conf_matrix = confusion_matrix(out, labels)  # create beautiful maps.... maybe
        self.matrix += conf_matrix
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        plot_confusion(self.matrix)
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _, targs = torch.max(labels, dim=1)
    return torch.tensor(torch.sum(preds == targs).item() / len(preds))


def confusion_matrix(outputs, labels):
    nb_classes = config.NUM_CLS
    matrix = torch.zeros(nb_classes, nb_classes)
    _, preds = torch.max(outputs, dim=1)
    _, targs = torch.max(labels, dim=1)
    for t, p in zip(targs.view(-1), preds.view(-1)):
        matrix[t.long(), p.long()] += 1
    return matrix


def plot_confusion(conf_matrix):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.set(rc={'figure.figsize': (600, 600)})
    ax[0] = sns.heatmap(conf_matrix, annot=True, linewidths=0.5,
                        ax=ax[0])
    plt.xlabel('Predicted classes')
    plt.ylabel('True classes')
    logger.info('accuracy: {}'.format(torch.sum(torch.tensor(np.diag(conf_matrix))) / torch.sum(conf_matrix)))
    print("accuracy:", torch.sum(torch.tensor(np.diag(conf_matrix))) / torch.sum(conf_matrix))
    cm = np.copy(conf_matrix)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax[1] = sns.heatmap(cmn, annot=True, linewidths=.5,
                        ax=ax[1])
    plt.xlabel('Predicted classes')
    plt.ylabel('True classes')
    fig.savefig('Conf_matrix/conf_aug_rot.png')

    tp = np.diag(conf_matrix)
    prec = list(map(truediv, tp, torch.sum(conf_matrix, dim=0)))
    rec = list(map(truediv, tp, torch.sum(conf_matrix, dim=1)))
    tp2 = list(np.multiply(prec, rec))
    s = [i + j for i, j in zip(prec, rec)]
    tp2 = [2 * i for i in tp2]
    f1 = list(map(truediv, tp2, s))
    f1_macro = sum(f1) / config.NUM_CLS
    print('Precision: {}\nRecall: {}\nF1 score: {}\nF1-macro: {}'.format(prec, rec, f1, f1_macro))
    logger.info('Precision: {}\nRecall: {}\nF1 score: {}\nF1-macro: {}'.format(prec, rec, f1, f1_macro))
