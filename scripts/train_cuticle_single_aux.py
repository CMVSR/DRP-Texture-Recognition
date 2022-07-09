"""Train DRP model on cuticle dataset."""

from dataclasses import dataclass
import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cuticulus.datasets import RoughSmoothFull

from drp.models.single_aux import myModel
from drp.utils import train_model

model_path = Path('./models/single_layer_with_aux/model.pkl')

INPUT_SIZE = 224
NUM_EPOCHS = 100
BATCH_SIZE = 16


class TorchDS(torch.utils.data.Dataset):
    """Torch dataset class for ant image dataset."""

    def __init__(
        self,
        imgs: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        """Initialize dataset.

        Args:
            imgs (np.ndarray): List of data.
            labels (np.ndarray): List of labels.
        """
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return length of dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx) -> tuple:
        """Return item at index idx.

        Returns:
            tuple: Tuple of image and label.
        """
        return self.imgs[idx], self.labels[idx]


def train_cuticle(device, pretrained: bool):
    # dataset setup
    ds = RoughSmoothFull(size=(INPUT_SIZE, INPUT_SIZE))
    ds.images = ds.images.transpose(0, 3, 1, 2)
    ds.images = ds.images.astype(np.float32)

    # split into train, test, and validation
    ds.stratified_split(n_samples=800)
    ds.split_validation()

    train_images, train_labels = ds.train()
    test_images, test_labels = ds.test()
    val_images, val_labels = ds.validate()

    # model setup
    model = myModel(num_classes=2)
    if pretrained:
        # load pretrained model
        pretrained_dict = torch.load(model_path).state_dict()

        filtered_layers = [
            'classifier.weight',
            'classifier.bias',
            'classifier_aux.weight',
            'classifier_aux.bias',
        ]
        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model.state_dict() and k not in filtered_layers}
        model.state_dict().update(filtered_dict)
        model.load_state_dict(model.state_dict())

    # init model
    model = model.to(device)
    params_to_update = model.parameters()
    # print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # print("\t", name)

    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # train model
    train_loader = torch.utils.data.DataLoader(
        dataset=TorchDS(train_images, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=TorchDS(val_images, val_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=TorchDS(test_images, test_labels),
        batch_size=int(BATCH_SIZE/2),
        shuffle=True,
    )
    model, val_acc_history, tra_acc_history, best_acc = train_model(
        model,
        dataloaders={
            'train': train_loader,
            'val': val_loader,
        },
        criterion=criterion,
        optimizer=optimizer_ft,
        num_epochs=NUM_EPOCHS,
        device=device,
    )

    # run model on test set
    pred_labels = []
    gt_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        pred_labels.extend(preds.tolist())
        gt_labels.extend(labels.tolist())

    # save model
    out_path = model_path.parent
    if pretrained:
        out_path = out_path / 'pretrained'
    else:
        out_path = out_path / 'not_pretrained'
    out_path = out_path / '{0}'.format(int(time.time()))
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(model, out_path / 'model.pkl')

    # save history
    tra_acc = [tra_acc_history[i].cpu().item()
               for i in range(len(tra_acc_history))]
    val_acc = [val_acc_history[i].cpu().item()
               for i in range(len(val_acc_history))]

    hist = {
        'val_acc_history': val_acc,
        'tra_acc_history': tra_acc,
    }
    with open(out_path / 'history.json', 'w') as fout:
        json.dump(hist, fout)

    # save test results
    res = {
        'pred_labels': pred_labels,
        'gt_labels': gt_labels,
    }
    with open(out_path / 'results.json', 'w') as fout:
        json.dump(res, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DRP on cuticle dataset.',
    )
    parser.add_argument(
        '--pretrained',
        type=bool,
        default=False,
        help='Whether to use pretrained model.',
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device to use for training.',
        required=True,
    )

    args = parser.parse_args()
    if args.pretrained:
        print('Using pretrained model from {0}.'.format(model_path))
    else:
        print('Not using pretrained model.')
    while True:
        train_cuticle('cuda:{0}'.format(args.device), args.pretrained)
