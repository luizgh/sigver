import argparse
import pathlib
from collections import OrderedDict

import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchvision import transforms
from visdom_logger.logger import VisdomLogger

import sigver.datasets.util as util
from sigver.featurelearning.data import TransformDataset
import sigver.featurelearning.models as models


def train(base_model: torch.nn.Module,
          classification_layer: torch.nn.Module,
          forg_layer: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          device: torch.device,
          callback: Optional[VisdomLogger],
          args: Any,
          logdir: Optional[pathlib.Path]):
    """ Trains a network using either SigNet or SigNet-F loss functions on
    https://arxiv.org/abs/1705.05787 (e.q. (1) and (4) on the paper)

    Parameters
    ----------
    base_model: torch.nn.Module
        The model architecture that "extract features" from signatures
    classification_layer: torch.nn.Module
        The classification layer (from features to predictions of which user
        wrote the signature)
    forg_layer: torch.nn.Module
        The forgery prediction layer (from features to predictions of whether
        the signature is a forgery). Only used in args.forg = True
    train_loader: torch.utils.data.DataLoader
        Iterable that loads the training set (x, y) tuples
    val_loader: torch.utils.data.DataLoader
        Iterable that loads the validation set (x, y) tuples
    device: torch.device
        The device (CPU or GPU) to use for training
    callback: VisdomLogger (optional)
        A callback to report the training progress
    args: Namespace
        Extra arguments for training: epochs, lr, lr_decay, lr_decay_times, momentum, weight_decay
    logdir: str
        Where to save the model and training curves

    Returns
    -------
    Dict (str -> tensors)
        The trained weights

    """

    # Collect all parameters that need to be optimizer
    parameters = list(base_model.parameters()) + list(classification_layer.parameters())
    if args.forg:
        parameters.extend(forg_layer.parameters())

    # Initialize optimizer and learning rate scheduler
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                          nesterov=True, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             args.epochs // args.lr_decay_times,
                                             args.lr_decay)

    best_acc = 0
    best_params = get_parameters(base_model, classification_layer, forg_layer)

    for epoch in range(args.epochs):
        # Train one epoch; evaluate on validation
        train_epoch(train_loader, base_model, classification_layer, forg_layer,
                    epoch, optimizer, lr_scheduler, callback, device, args)

        val_metrics = test(val_loader, base_model, classification_layer, device, args.forg, forg_layer)
        val_acc, val_loss, val_forg_acc, val_forg_loss = val_metrics

        # Save the best model only on improvement (early stopping)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_params = get_parameters(base_model, classification_layer, forg_layer)
            if logdir is not None:
                torch.save(best_params, logdir / 'model_best.pth')

        if callback:
            callback.scalar('val_loss', epoch + 1, val_loss)
            callback.scalar('val_acc', epoch + 1, val_acc)

            if args.forg:
                callback.scalar('val_forg_loss', epoch + 1, val_forg_loss)
                callback.scalar('val_forg_acc', epoch + 1, val_forg_acc)

        if args.forg:
            print('Epoch {}. Val loss: {:.4f}, Val acc: {:.2f}%,'
                  'Val forg loss: {:.4f}, Val forg acc: {:.2f}%'.format(epoch, val_loss,
                                                                        val_acc * 100,
                                                                        val_forg_loss,
                                                                        val_forg_acc * 100))
        else:
            print('Epoch {}. Val loss: {:.4f}, Val acc: {:.2f}%'.format(epoch, val_loss, val_acc * 100))

        if logdir is not None:
            current_params = get_parameters(base_model, classification_layer, forg_layer)
            torch.save(current_params, logdir / 'model_last.pth')
            if callback:
                callback.save(logdir / 'train_curves.pickle')

    return best_params


def copy_to_cpu(weights: Dict[str, Any]):
    return OrderedDict([(k, v.cpu()) for k, v in weights.items()])


def get_parameters(base_model, classification_layer, forg_layer):
    best_params = (copy_to_cpu(base_model.state_dict()),
                   copy_to_cpu(classification_layer.state_dict()),
                   copy_to_cpu(forg_layer.state_dict()))
    return best_params


def train_epoch(train_loader: torch.utils.data.DataLoader,
                base_model: torch.nn.Module,
                classification_layer: torch.nn.Module,
                forg_layer: torch.nn.Module,
                epoch: int,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                callback: Optional[VisdomLogger],
                device: torch.device,
                args: Any):
    """ Trains the network for one epoch

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            Iterable that loads the training set (x, y) tuples
        base_model: torch.nn.Module
            The model architecture that "extract features" from signatures
        classification_layer: torch.nn.Module
            The classification layer (from features to predictions of which user
            wrote the signature)
        forg_layer: torch.nn.Module
            The forgery prediction layer (from features to predictions of whether
            the signature is a forgery). Only used in args.forg = True
        epoch: int
            The current epoch (used for reporting)
        optimizer: torch.optim.Optimizer
            The optimizer (already initialized)
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler
        callback: VisdomLogger (optional)
            A callback to report the training progress
        device: torch.device
            The device (CPU or GPU) to use for training
        args: Namespace
            Extra arguments used for training:
            args.forg: bool
                Whether forgeries are being used for training
            args.lamb: float
                The weight used for the forgery loss (training with forgeries only)

        Returns
        -------
        None
        """

    step = 0
    n_steps = len(train_loader)
    for batch in train_loader:
        x, y = batch[0], batch[1]
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        yforg = torch.tensor(batch[2], dtype=torch.float).to(device)

        # Forward propagation
        features = base_model(x)

        if args.forg:
            if args.loss_type == 'L1':
                # Eq (3) in https://arxiv.org/abs/1705.05787
                logits = classification_layer(features)
                class_loss = F.cross_entropy(logits, y)

                forg_logits = forg_layer(features).squeeze()
                forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)

                loss = (1 - args.lamb) * class_loss
                loss += args.lamb * forg_loss
            else: 
                # Eq (4) in https://arxiv.org/abs/1705.05787
                logits = classification_layer(features[yforg == 0])
                class_loss = F.cross_entropy(logits, y[yforg == 0])

                forg_logits = forg_layer(features).squeeze()
                forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)

                loss = (1 - args.lamb) * class_loss
                loss += args.lamb * forg_loss
        else:
            # Eq (1) in https://arxiv.org/abs/1705.05787
            logits = classification_layer(features)
            loss = class_loss = F.cross_entropy(logits, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10)

        # Update weights
        optimizer.step()

        # Logging
        if callback and step % 100 == 0:
            iteration = epoch + (step / n_steps)
            callback.scalar('class_loss', iteration, class_loss.detach())

            pred = logits.argmax(1)
            if args.loss_type == 'L1': acc = y.eq(pred).float().mean()
            else: acc = y[yforg == 0].eq(pred[yforg == 0]).float().mean()
            callback.scalar('train_acc', epoch + (step / n_steps), acc.detach())
            if args.forg:
                forg_pred = forg_logits > 0
                forg_acc = yforg.long().eq(forg_pred.long()).float().mean()
                callback.scalar('forg_loss', iteration, forg_loss.detach())
                callback.scalar('forg_acc', iteration, forg_acc.detach())

        step += 1
    lr_scheduler.step()


def test(val_loader: torch.utils.data.DataLoader,
         base_model: torch.nn.Module,
         classification_layer: torch.nn.Module,
         device: torch.device,
         is_forg: bool,
         forg_layer: Optional[torch.nn.Module] = None) -> Tuple[float, float, float, float]:
    """ Test the model in a validation/test set

    Parameters
    ----------
    val_loader: torch.utils.data.DataLoader
        Iterable that loads the validation set (x, y) tuples
    base_model: torch.nn.Module
        The model architecture that "extract features" from signatures
    classification_layer: torch.nn.Module
        The classification layer (from features to predictions of which user
        wrote the signature)
    device: torch.device
        The device (CPU or GPU) to use for training
    is_forg: bool
        Whether or not forgeries are being used for training/testing
    forg_layer: torch.nn.Module
            The forgery prediction layer (from features to predictions of whether
            the signature is a forgery). Only used in is_forg = True

    Returns
    -------
    float, float
        The valication accuracy and validation loss

    """
    val_losses = []
    val_accs = []

    val_forg_losses = []
    val_forg_accs = []
    for batch in val_loader:
        x, y, yforg = batch[0], batch[1], batch[2]
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        yforg = torch.tensor(yforg, dtype=torch.float).to(device)

        with torch.no_grad():
            features = base_model(x)
            logits = classification_layer(features[yforg == 0])

            loss = F.cross_entropy(logits, y[yforg == 0])
            pred = logits.argmax(1)
            acc = y[yforg == 0].eq(pred).float().mean()

            if is_forg:
                forg_logits = forg_layer(features).squeeze()
                forg_loss = F.binary_cross_entropy_with_logits(forg_logits, yforg)
                forg_pred = forg_logits > 0
                forg_acc = yforg.long().eq(forg_pred.long()).float().mean()

                val_forg_losses.append(forg_loss.item())
                val_forg_accs.append(forg_acc.item())

        val_losses.append(loss.item())
        val_accs.append(acc.item())
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_forg_loss = np.mean(val_forg_losses) if len(val_forg_losses) > 0 else np.nan
    val_forg_acc= np.mean(val_forg_accs) if len(val_forg_accs) > 0 else np.nan

    return val_acc, val_loss, val_forg_acc, val_forg_loss


def main(args):
    # Setup logging
    logdir = pathlib.Path(args.logdir)
    if not logdir.exists():
        logdir.mkdir()

    if args.visdomport is not None:
        logger = VisdomLogger(port=args.visdomport)
    else:
        logger = None

    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: {}'.format(device))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print('Loading Data')

    x, y, yforg, usermapping, filenames = util.load_dataset(args.dataset_path)
    data = util.get_subset((x, y, yforg), subset=range(*args.users))
    if not args.forg:
        data = util.remove_forgeries(data, forg_idx=2)

    train_loader, val_loader = setup_data_loaders(data, args.batch_size, args.input_size)

    print('Initializing Model')

    n_classes = len(np.unique(data[1]))

    base_model = models.available_models[args.model]().to(device)
    classification_layer = nn.Linear(base_model.feature_space_size, n_classes).to(device)
    if args.forg:
        forg_layer = nn.Linear(base_model.feature_space_size, 1).to(device)
    else:
        forg_layer = nn.Module()  # Stub module with no parameters

    if args.test:
        print('Testing')
        base_model_params, classification_params, forg_params = torch.load(args.checkpoint)
        base_model.load_state_dict(base_model_params)

        classification_layer.load_state_dict(classification_params)
        if args.forg:
            forg_layer.load_state_dict(forg_params)
        val_acc, val_loss, val_forg_acc, val_forg_loss = test(val_loader, base_model, classification_layer,
                                                              device, args.forg, forg_layer)
        if args.forg:
            print('Val loss: {:.4f}, Val acc: {:.2f}%,'
                  'Val forg loss: {:.4f}, Val forg acc: {:.2f}%'.format(val_loss,
                                                                        val_acc * 100,
                                                                        val_forg_loss,
                                                                        val_forg_acc * 100))
        else:
            print('Val loss: {:.4f}, Val acc: {:.2f}%'.format(val_loss, val_acc * 100))

    else:
        print('Training')
        train(base_model, classification_layer, forg_layer, train_loader, val_loader,
              device, logger, args, logdir)


def setup_data_loaders(data, batch_size, input_size):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[1])
    data = TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(y), torch.from_numpy(data[2]))
    train_size = int(0.9 * len(data))
    sizes = (train_size, len(data) - train_size)
    train_set, test_set = random_split(data, sizes)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
    ])
    train_set = TransformDataset(train_set, train_transforms)
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    test_set = TransformDataset(test_set, val_transforms)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Train Signet/F')
    argparser.add_argument('--dataset-path', help='Path containing a numpy file with images and labels')
    argparser.add_argument('--input-size', help='Input size (cropped)', nargs=2, type=int, default=(150, 220))
    argparser.add_argument('--users', nargs=2, type=int, default=(350, 881))

    argparser.add_argument('--model', help='Model architecture', choices=models.available_models, required=True)
    argparser.add_argument('--batch-size', help='Batch size', type=int, default=32)
    argparser.add_argument('--lr', help='learning rate', default=0.001, type=float)
    argparser.add_argument('--lr-decay', help='learning rate decay (multiplier)', default=0.1, type=float)
    argparser.add_argument('--lr-decay-times', help='number of times learning rate decays', default=3, type=float)
    argparser.add_argument('--momentum', help='momentum', default=0.90, type=float)
    argparser.add_argument('--weight-decay', help='Weight Decay', default=1e-4, type=float)
    argparser.add_argument('--epochs', help='Number of epochs', default=20, type=int)
    argparser.add_argument('--checkpoint', help='starting weights (pth file)')
    argparser.add_argument('--test', action='store_true')

    argparser.add_argument('--seed', default=42, type=int)

    argparser.add_argument('--forg', dest='forg', action='store_true')
    argparser.add_argument('--lamb', type=float, help='Lambda for trading of user classification '
                                                      'and forgery classification')
    argparser.add_argument('--loss-type', help='L1 or L2 loss, implemented on paper Eq(3) or Eq(4)', default='L2', type=str)

    argparser.add_argument('--gpu-idx', default=0, type=int)
    argparser.add_argument('--logdir', help='logdir', required=True)
    argparser.add_argument('--visdomport', help='Visdom port (plotting)', type=int)

    argparser.set_defaults(forg=False, test=False)
    arguments = argparser.parse_args()
    print(arguments)

    main(arguments)
