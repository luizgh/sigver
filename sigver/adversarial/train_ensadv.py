import argparse
import pathlib
from collections import OrderedDict

import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable, List
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
          adv_models: List[torch.nn.Module],
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
        train_epoch(train_loader, base_model, classification_layer, forg_layer, adv_models,
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


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


def create_adversarial(adv_models, adv_model_idx, x, y, eps):
    adv_model = adv_models[adv_model_idx]
    adv_model.eval()
    x.requires_grad_(True)
    output_other_model = adv_model(x)
    loss = F.cross_entropy(output_other_model, y)
    grad = torch.autograd.grad(loss, x)[0]
    adv = x + eps * grad / l2_norm(grad).view(-1, 1, 1, 1)
    adv.clamp_(0, 1)
    x.requires_grad_(False)

    adv_model.train()

    return adv.detach()


def train_epoch(train_loader: torch.utils.data.DataLoader,
                base_model: torch.nn.Module,
                classification_layer: torch.nn.Module,
                forg_layer: torch.nn.Module,
                adv_models: List[torch.nn.Module],
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

    adv_model_idx = 0
    for batch in train_loader:
        x, y = batch[0], batch[1]
        x = torch.tensor(x, dtype=torch.float).to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        yforg = torch.tensor(batch[2], dtype=torch.float).to(device)

        # Create adversarial example
        adv = create_adversarial(adv_models, adv_model_idx, x, y, args.eps)

        # Clean example
        features = base_model(x)

        if args.forg:
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
        loss = args.alpha * loss
        optimizer.zero_grad()
        loss.backward()

        # adv example
        adv_features = base_model(adv)
        adv_logits = classification_layer(adv_features)
        adv_loss = F.cross_entropy(adv_logits, y)
        loss2 = (1 - args.alpha) * adv_loss
        loss2.backward()

        torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'], 10)

        # Update weights
        optimizer.step()

        # Logging
        if callback and step % 11 == 0:
            with torch.no_grad():
                pred_clean = logits.argmax(1)
                acc_clean = y[yforg == 0].eq(pred_clean).float().mean()

                pred_adv = adv_logits.argmax(1)
                acc_adv = y[yforg == 0].eq(pred_adv).float().mean()

            iteration = epoch + (step / n_steps)
            callback.scalars(['closs_clean', 'closs_adv'], iteration, [class_loss.detach(), adv_loss.detach()])
            callback.scalar('closs_adv_{}'.format(adv_model_idx), iteration,
                            adv_loss.detach())
            callback.scalars(['acc_clean', 'acc_addv'], epoch + (step / n_steps), [acc_clean, acc_adv.detach()])
            if args.forg:
                forg_pred = forg_logits > 0
                forg_acc = yforg.long().eq(forg_pred.long()).float().mean()
                callback.scalar('forg_loss', iteration, forg_loss.detach())
                callback.scalar('forg_acc', iteration, forg_acc.detach())

        step += 1
        adv_model_idx = (adv_model_idx + 1) % len(adv_models)
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

                val_forg_losses.append(forg_loss)
                val_forg_accs.append(forg_acc)

        val_losses.append(loss.item())
        val_accs.append(acc.item())
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_forg_loss = np.mean(val_forg_losses).item() if len(val_forg_losses) > 0 else np.nan
    val_forg_acc= np.mean(val_forg_accs).item() if len(val_forg_accs) > 0 else np.nan

    return val_acc.item(), val_loss.item(), val_forg_acc, val_forg_loss


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

    model = nn.Sequential(base_model, classification_layer)
    adv_models = [model]

    if args.trained_models is not None:
        assert len(
            args.trained_models) % 2 == 0  # Should be pairs of <model> <model_path>
        for i in range(0, len(args.trained_models), 2):
            model_name = args.trained_models[i]
            all_params = torch.load(args.trained_models[i + 1])
            adv_params, adv_classification_params, _ = all_params

            adv_base_model = models.available_models[model_name]().to(device)
            adv_base_model.load_state_dict(adv_params)
            adv_clf_layer = nn.Linear(adv_base_model.feature_space_size,
                                      n_classes).to(device)
            adv_clf_layer.load_state_dict(adv_classification_params)

            adv_model = nn.Sequential(adv_base_model, adv_clf_layer)

            adv_models.append(adv_model)

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

        train(base_model, classification_layer, forg_layer, adv_models, train_loader, val_loader,
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

    argparser.add_argument('--trained-models', nargs='*',
                           help='Trained models: <model> <model_path> [<model> <model_path>]')

    argparser.add_argument('--gpu-idx', default=0, type=int)
    argparser.add_argument('--logdir', help='logdir', required=True)
    argparser.add_argument('--visdomport', help='Visdom port (plotting)', type=int)

    argparser.add_argument('--alpha', default=0.5,
                           help='Trade-off between loss and adv_loss')
    argparser.add_argument('--eps', default=5, type=int,
                           help='L2 norm of the adversarial images')

    argparser.set_defaults(forg=False, test=False)
    arguments = argparser.parse_args()
    print(arguments)

    main(arguments)
