import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from operator import truediv


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    # rewrite
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        temp_labels = labels.softmax(dim=1)
        loss = F.cross_entropy(out, temp_labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        conf_matrix = confusion_matrix(out, labels)  # create beautiful maps.... maybe
        plot_confusion(conf_matrix)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
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
    nb_classes = 15
    matrix = torch.zeros(nb_classes, nb_classes)
    _, preds = torch.max(outputs, dim=1)
    _, targs = torch.max(labels, dim=1)
    for t, p in zip(targs.view(-1), preds.view(-1)):
        matrix[t.long(), p.long()] += 1
    return matrix


def plot_confusion(conf_matrix):
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    ax[0] = sns.heatmap(conf_matrix, annot=True, linewidths=.5,
                        ax=ax[0])
    print("accuracy:", np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))
    fig.show()

    cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    ax[1] = sns.heatmap(cmn, annot=True, linewidths=.5,
                        ax=ax[1])
    fig.show()

    tp = np.diag(conf_matrix)
    prec = list(map(truediv, tp, np.sum(conf_matrix, axis=0)))
    rec = list(map(truediv, tp, np.sum(conf_matrix, axis=1)))
    print('Precision: {}\nRecall: {}'.format(prec, rec))
