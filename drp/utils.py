"""Utilities for the DRP model."""

import time
import copy

import torch


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device: torch.device,
    num_epochs: int = 25,
    is_inception: bool = False,
):
    since = time.time()

    val_acc_history = []
    tra_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to train mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)  # outputs_aux
                    # +0.2*criterion(outputs_aux, labels)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in train it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in val we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        # print(inputs.shape)
                        outputs = model(inputs)
                        # +0.2*criterion(outputs_aux, labels)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step(closure)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                tra_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, tra_acc_history, best_acc


def set_parameter_requires_grad(
    model: torch.nn.Module,
    feature_extracting: bool = False,
):
    """Function for deciding which part is required to be updated

    Args:
        model (torch.nn.Module): The model to be updated.
        feature_extracting (bool): Whether to update the layers or not.
    """
    # Flag for feature extracting. When False, we finetune the whole model,
    # when True we only update the reshaped layer params
    # feature_extract = True
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
