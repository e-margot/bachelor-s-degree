# -*- coding: utf-8 -*-

import torch
from torchvision.transforms import PILToTensor, ConvertImageDtype, Compose, RandomRotation, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import torchvision
from PIL import Image
from data_classes.my_coco_collection import MyCocoDetection
from data_classes.device_data_loader import DeviceDataLoader
from model_classes.flir_resnet import FLIR80Resnet
from model_classes.resnet import ResNet, BasicBlock
from model_classes.image_classificaton_base import ImageClassificationBase
from data_classes.augmentate_coco import AugMyCocoDetection
import config
from loguru import logger

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


@torch.no_grad()
def evaluate(this_model, val_loader):
    this_model.eval()
    outputs2 = []
    for batch in val_loader:
        outputs2.append(this_model.validation_step(batch))
    return this_model.validation_epoch_end(outputs2)


def fit(epochs, lr, this_model, train_loader, val_loader,
        weight_decay=0, this_opt_func=torch.optim.SGD):
    loss_history = []
    logger.info("Upload model weights")
    if os.path.isfile('weights/model_aug.pth'):
        this_model.load_state_dict(torch.load('weights/model_aug.pth',  map_location=torch.device('cpu')))
    optimizer = this_opt_func(this_model.parameters(), lr, weight_decay=weight_decay)
    logger.info("Start training cycle")

    for epoch in range(epochs):
        logger.info(f"Running {epoch} epoch")
        # Training Phase
        this_model.train()
        train_losses = []
        for batch in train_loader:
            loss = this_model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(this_model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        this_model.epoch_end(epoch, result)
        loss_history.append(result)
        torch.save(this_model.state_dict(), 'weights/model.pth')
    return loss_history


def plot_accuracies(acc_history, file_name):
    accuracies = [x['val_acc'] for x in acc_history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(file_name)

def plot_losses(history, file_name):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(file_name)

# def plot_losses(loss_history, file_name):
#     train_losses = [x.get('train_loss') for x in loss_history]
#     val_losses = [x['val_loss'] for x in loss_history]
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.title('Loss vs. No. of epochs')
#     plt.savefig(file_name)


losses_names = {
    1: 'loss_func1.jpg',
    2: 'loss_func2.jpg',
    3: 'loss_func3.jpg',
    4: 'loss_func4.jpg',
}

accuracy_names = {
    1: 'accuracy1.jpg',
    2: 'accuracy2.jpg',
    3: 'accuracy3.jpg',
    4: 'accuracy4.jpg',
}


def aug_my_image(image: Image.Image) -> Image.Image:
    im_aug = torchvision.transforms.RandomInvert(p=0.67)(image)
    logger.info("AUG: RandomInvert(0.67)")
    im_aug = torchvision.transforms.functional.adjust_sharpness(im_aug, 10)
    logger.info("AUG: adjust_sharpness(10)")
    return im_aug

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thermal classification')
    parser.add_argument('-e', dest='epoch', type=int, default=10, help='Enter number of epoch')
    args = parser.parse_args()
    logger.add("logs/logs_aug.log", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    # Download dataset
    logger.info("Preparing data")
    logger.info("AUG: RandomRotation(15)")
    transform_rgb = Compose([PILToTensor(),
                             ConvertImageDtype(torch.float),
                             RandomRotation(15),
                             Lambda(aug_my_image),
                             ])
    transform_thermal = Compose([PILToTensor(),
                                 ConvertImageDtype(torch.float)
                                 ])

    batch_size = config.BATCH_SIZE

    # TV

    path2data = config.TV_PATH_TO_DATA
    path2json = config.TV_PATH_TO_JSON

    coco_TV = MyCocoDetection(path2data, path2json, transform=transform_rgb)
    coco_loader_TV = DataLoader(coco_TV, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # NIR

    path2data = config.NIR_PATH_TO_DATA
    path2json = config.NIR_PATH_TO_JSON

    coco_NIR = MyCocoDetection(path2data, path2json, transform=transform_thermal)
    coco_loader_NIR = DataLoader(coco_NIR, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    device = get_default_device()
    logger.info(f"Using device:{device}")

    coco_loader_NIR = DeviceDataLoader(coco_loader_NIR, device)
    coco_loader_TV = DeviceDataLoader(coco_loader_TV, device)
    logger.info("DataLoaders is ready")
    logger.info("Creating model")

    model = FLIR80Resnet()

    model = model.to(device)

    # For this model we gonna use Adam Optimization
    opt_func = torch.optim.Adam

    train_dl = coco_loader_TV
    valid_dl = coco_loader_NIR
    epoch = args.epoch

    training_cycles = {
        1: [epoch, 1e-2, model, train_dl, valid_dl, 5e-4, opt_func],
        2: [epoch, 1e-3, model, train_dl, valid_dl, 5e-4, opt_func],
        3: [epoch, 1e-4, model, train_dl, valid_dl, 5e-4, opt_func],
        4: [epoch, 1e-5, model, train_dl, valid_dl, 5e-4, opt_func],
        5: [epoch, 1e-6, model, train_dl, valid_dl, 5e-4, opt_func]
    }

    INDEX = config.RUN_RULE_SET
    logger.info(f"Using rule set {config.RUN_RULE_SET}")
    history = fit(*training_cycles[INDEX])
    plot_accuracies(history, accuracy_names[INDEX])
    plot_losses(history, losses_names[INDEX])
