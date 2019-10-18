import pickle

import torch
import numpy as np
import argparse
import pathlib

from tqdm import tqdm
from visdom_logger import VisdomLogger
import sigver.metalearning.models as models
from sigver.metalearning.maml_pytorch import MAML, get_per_step_loss_importance_vector, balanced_binary_cross_entropy
import sigver.featurelearning.train as pretrain
from sigver.datasets.util import load_dataset
from sigver.datasets import util
from sigver.metalearning.models.pretrain_wrapper import PretrainWrapper
import torch.nn as nn
from sigver.metalearning.data import MAMLDataSet
from torch.utils.data import DataLoader
from sigver.wd.metrics import compute_metrics

argparser = argparse.ArgumentParser()
argparser.add_argument('--original-dataset-path', help='Path containing signature images')
argparser.add_argument('--dataset-path', help='Path containing a numpy file with images and labels')
argparser.add_argument('--img-size', help='Image size', nargs=2, type=int, default=(170, 242))
argparser.add_argument('--input-size', help='Input size (cropped)', nargs=2, type=int, default=(150, 220))

argparser.add_argument('--num-gen', help='Number of genuine signatures for train', default=5, type=int)
argparser.add_argument('--num-rf', help='Number of Random forgeries for train', default=10, type=int)
argparser.add_argument('--num-gen-test', help='Number of genuine signatures for test', default=10, type=int)
argparser.add_argument('--num-rf-test',
                       help='Number of random forgeries for test',
                       default=10, type=int)
argparser.add_argument('--num-sk-test',
                       help='Number of skilled forgeries for test', required=True, type=int)

argparser.add_argument('--model', help='Model architecture', choices=models.available_models, required=True)
argparser.add_argument('--train-lr', help='task learning rate', default=1e-3, type=float)
argparser.add_argument('--learn-task-lr', help='learn task lr (per step per layer)',
                       action='store_true')
argparser.add_argument('--meta-batch-size', help='meta batch size', default=4, type=int)
argparser.add_argument('--meta-lr', help='meta learning rate', default=1e-3, type=float)
argparser.add_argument('--meta-min-lr', help='minimum meta learning rate (after annealing)',
                       default=1e-5, type=float)
argparser.add_argument('--num-updates', help='Number of updates (gradient descent iterations)',
                       required=True, type=int)
argparser.add_argument('--epochs', help='Number of epochs', default=100, type=int)
argparser.add_argument('--pretrain-epochs', help='Number of train epochs', default=0, type=int)
argparser.add_argument('--pretrain-forg', action='store_true', dest='pretrain_forg')
argparser.add_argument('--pretrain-forg-lambda', type=float, default=0.95)
argparser.add_argument('--msl-epochs', help='Number of train epochs with multi-step-loss',
                       default=15, type=int)

argparser.add_argument('--gpu-idx', type=int, default=0)

argparser.add_argument('--logdir', help='logdir')
argparser.add_argument('--test', dest='test', action='store_true')
argparser.add_argument('--use-testset', dest='use_testset', action='store_true')
argparser.add_argument('--dev-users', nargs=2, type=int, default=(350, 881),
                       help='Users for meta-training')

argparser.add_argument('--exp-users', nargs=2, type=int,
                       help='Users to test the system with. '
                            '(overrides "use-testset")')

argparser.add_argument('--devset-size', type=int,
                       help='Number of users in the devset (meta-train)')
argparser.add_argument('--devset-sk-size', type=int,
                       help='Number of users in the devset (meta-train)')

argparser.add_argument('--checkpoint', help='(pre) trained model to initialize'
                                            ' the weights')
argparser.add_argument('--seed', type=int, default=1234)
argparser.add_argument('--folds', type=int, default=10)
argparser.add_argument('--save-file', help='Filename to save results on the '
                                           'validation/test set')


argparser.set_defaults(test=False, use_testset=False, pretrain_forg=False)

argparser.add_argument('--port', help='Visdom port (plotting)', type=int)


def get_logdir(args):
    path = 'gen{}_rf{}_tgen{}_trf{}_tsk{}_pretrain{}_epochs{}_nupdates{}_devsize{}_devsksize{}_tasklr_{}'.format(args.num_gen,
        args.num_rf, args.num_gen_test, args.num_rf_test, args.num_sk_test, args.pretrain_epochs, args.epochs,
        args.num_updates, args.devset_size, args.devset_sk_size, args.train_lr)
    return pathlib.Path('~/runs/').expanduser() / path


def main(args):
    rng = np.random.RandomState(args.seed)

    if args.test:
        assert args.checkpoint is not None, 'Please inform the checkpoint (trained model)'

    if args.logdir is None:
        logdir = get_logdir(args)
    else:
        logdir = pathlib.Path(args.logdir)
    if not logdir.exists():
        logdir.mkdir()

    print('Writing logs to {}'.format(logdir))

    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    if args.port is not None:
        logger = VisdomLogger(port=args.port)
    else:
        logger = None

    print('Loading Data')
    x, y, yforg, usermapping, filenames = load_dataset(args.dataset_path)

    dev_users = range(args.dev_users[0], args.dev_users[1])
    if args.devset_size is not None:
        # Randomly select users from the dev set
        dev_users = rng.choice(dev_users, args.devset_size, replace=False)

    if args.devset_sk_size is not None:
        assert args.devset_sk_size <= len(dev_users), 'devset-sk-size should be smaller than devset-size'

        # Randomly select users from the dev set to have skilled forgeries (others don't)
        dev_sk_users = set(rng.choice(dev_users, args.devset_sk_size, replace=False))
    else:
        dev_sk_users = set(dev_users)

    print('{} users in dev set; {} users with skilled forgeries'.format(
        len(dev_users), len(dev_sk_users)
    ))

    if args.exp_users is not None:
        val_users = range(args.exp_users[0], args.exp_users[1])
        print('Testing with users from {} to {}'.format(args.exp_users[0],
                                                        args.exp_users[1]))
    elif args.use_testset:
        val_users = range(0, 300)
        print('Testing with Exploitation set')
    else:
        val_users = range(300, 350)


    print('Initializing model')
    base_model = models.available_models[args.model]().to(device)
    weights = base_model.build_weights(device)
    maml = MAML(base_model, args.num_updates, args.num_updates, args.train_lr, args.meta_lr,
                args.meta_min_lr, args.epochs, args.learn_task_lr, weights, device, logger,
                loss_function=balanced_binary_cross_entropy, is_classification=True)

    if args.checkpoint:
        params = torch.load(args.checkpoint)
        maml.load(params)

    if args.test:
        test_and_save(args, device, logdir, maml, val_users, x, y, yforg)
        return

    # Pretraining
    if args.pretrain_epochs > 0:
        print('Pre-training')
        data = util.get_subset((x, y, yforg), subset=range(350, 881))

        wrapped_model = PretrainWrapper(base_model, weights)

        if not args.pretrain_forg:
            data = util.remove_forgeries(data, forg_idx=2)

        train_loader, val_loader = pretrain.setup_data_loaders(data, 32, args.input_size)
        n_classes = len(np.unique(y))

        classification_layer = nn.Linear(base_model.feature_space_size, n_classes).to(device)
        if args.pretrain_forg:
            forg_layer = nn.Linear(base_model.feature_space_size, 1).to(device)
        else:
            forg_layer = nn.Module()  # Stub module with no parameters

        pretrain_args = argparse.Namespace(lr=0.01, lr_decay=0.1, lr_decay_times=1,
                                           momentum=0.9, weight_decay=0.001, forg=args.pretrain_forg,
                                           lamb=args.pretrain_forg_lambda, epochs=args.pretrain_epochs)
        print(pretrain_args)
        pretrain.train(wrapped_model, classification_layer, forg_layer, train_loader, val_loader,
                       device, logger, pretrain_args, logdir=None)

    # MAML training

    trainset = MAMLDataSet(data=(x, y, yforg),
                           subset=dev_users,
                           sk_subset=dev_sk_users,
                           num_gen_train=args.num_gen,
                           num_rf_train=args.num_rf,
                           num_gen_test=args.num_gen_test,
                           num_rf_test=args.num_rf_test,
                           num_sk_test=args.num_sk_test,
                           input_shape=args.input_size,
                           test=False,
                           rng=np.random.RandomState(args.seed))

    val_set = MAMLDataSet(data=(x, y, yforg), subset=val_users, num_gen_train=args.num_gen, num_rf_train=args.num_rf,
                          num_gen_test=args.num_gen_test, num_rf_test=args.num_rf_test, num_sk_test=args.num_sk_test,
                          input_shape=args.input_size, test=True, rng=np.random.RandomState(args.seed))

    loader = DataLoader(trainset, batch_size=args.meta_batch_size,
                        shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    print('Training')
    best_val_acc = 0
    with tqdm(initial=0, total=len(loader) * args.epochs) as pbar:
        if args.checkpoint is not None:
            postupdate_accs, postupdate_losses, preupdate_losses = test_one_epoch(
                maml,
                val_set,
                device,
                args.num_updates)

            if logger:
                for i in range(args.num_updates):
                    logger.scalar('val_postupdate_loss_{}'.format(i), 0,
                                  np.mean(postupdate_losses, axis=0)[i])

                    logger.scalar('val_postupdate_acc_{}'.format(i), 0,
                                  np.mean(postupdate_accs, axis=0)[i])

        for epoch in range(args.epochs):
            loss_weights = get_per_step_loss_importance_vector(args.num_updates,
                                                               args.msl_epochs,
                                                               epoch)

            n_batches = len(loader)
            for step, item in enumerate(loader):
                item = move_to_gpu(*item, device=device)
                maml.meta_learning_step((item[0], item[1]),
                                        (item[2], item[3]),
                                        loss_weights,
                                        epoch + step / n_batches)
                pbar.update(1)

            maml.scheduler.step()

            postupdate_accs, postupdate_losses, preupdate_losses = test_one_epoch(maml,
                                                                                  val_set,
                                                                                  device,
                                                                                  args.num_updates)

            if logger:
                for i in range(args.num_updates):
                    logger.scalar('val_postupdate_loss_{}'.format(i), epoch+1, np.mean(postupdate_losses, axis=0)[i])

                    logger.scalar('val_postupdate_acc_{}'.format(i), epoch+1, np.mean(postupdate_accs, axis=0)[i])

                logger.save(logdir / 'train_curves.pickle')
            this_val_loss = np.mean(postupdate_losses, axis=0)[-1]
            this_val_acc = np.mean(postupdate_accs, axis=0)[-1]

            if this_val_acc > best_val_acc:
                best_val_acc = this_val_acc
                torch.save(maml.parameters, logdir / 'best_model.pth')
            print('Epoch {}. Val loss: {:.4f}. Val Acc: {:.2f}%'.format(epoch, this_val_loss, this_val_acc * 100))

    # Re-load best parameters and test with 10 folds
    params = torch.load(logdir / 'best_model.pth')
    maml.load(params)

    test_and_save(args, device, logdir, maml, val_users, x, y, yforg)


def test_and_save(args, device, logdir, maml, val_users, x, y, yforg):
    eer_u_list = []
    eer_list = []
    all_results = []
    val_set_rng = np.random.RandomState(args.seed)
    if args.save_file is None:
        save_path = logdir / 'results_{}_to_{}_samples_{}.pickle'.format(val_users[0], val_users[-1], args.num_gen)
    else:
        save_path = logdir / args.save_file
    for _ in tqdm(range(args.folds)):
        val_set = MAMLDataSet(data=(x, y, yforg), subset=val_users, num_gen_train=args.num_gen,
                              num_rf_train=args.num_rf, num_gen_test=args.num_gen_test, num_rf_test=args.num_rf_test,
                              num_sk_test=args.num_sk_test, input_shape=args.input_size, test=True, rng=val_set_rng)
        results = compute_val_metrics(args, device, maml, val_set)
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']

        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)
    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))
    print('Saving results to {}'.format(save_path))
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)


def compute_val_metrics(args, device, maml, val_set):
    gen_error = []
    rf_error = []
    sk_error = []
    all_gen_preds = []
    all_rf_preds = []
    all_sk_preds = []
    for item in val_set:
        train_x = torch.tensor(item[0], dtype=torch.float).to(device)
        train_y = torch.tensor(item[1], dtype=torch.float).to(device)
        test_x = torch.tensor(item[2], dtype=torch.float).to(device)
        test_y = torch.tensor(item[3], dtype=torch.float).to(device)
        test_yforg = torch.tensor(item[4], dtype=torch.float).to(device)

        fast_weight_list = maml.adapt_weights_to_task(train_x, train_y, args.num_updates, is_training=False)

        fast_weights = fast_weight_list[-1]

        with torch.no_grad():
            logits = maml.model.forward(test_x, fast_weights, training=False)

            gen_predictions = logits[test_y == 1]
            rf_predictions = logits[(test_y == 0) & (test_yforg == 0)]
            sk_predictions = logits[(test_y == 0) & (test_yforg == 1)]

            all_gen_preds.append(gen_predictions.cpu().numpy())
            all_rf_preds.append(rf_predictions.cpu().numpy())
            all_sk_preds.append(sk_predictions.cpu().numpy())

            gen_error.append((gen_predictions <= 0).type(torch.float).mean().item())
            rf_error.append((rf_predictions > 0).type(torch.float).mean().item())
            sk_error.append((sk_predictions > 0).type(torch.float).mean().item())

    m = compute_metrics(all_gen_preds, all_rf_preds, all_sk_preds, global_threshold=0)
    return {'all_metrics': m,
            'predictions': {'genuinePreds': all_gen_preds,
                            'randomPreds': all_rf_preds,
                            'skilledPreds': all_sk_preds,
                            }
            }


def test_one_epoch(maml, val_set, device, num_updates):
    preupdate_losses = []
    postupdate_losses = []
    postupdate_accs = []
    for item in val_set:
        train_x = torch.tensor(item[0], dtype=torch.float).to(device)
        train_y = torch.tensor(item[1], dtype=torch.float).to(device)
        test_x = torch.tensor(item[2], dtype=torch.float).to(device)
        test_y = torch.tensor(item[3], dtype=torch.float).to(device)


        fast_weight_list = maml.adapt_weights_to_task(
            train_x, train_y, num_updates, is_training=False)

        preupdate_loss, preupdate_acc = maml.test_on_task(maml.weights, test_x, test_y)

        losses = []
        accs = []
        for i in range(num_updates):
            with torch.no_grad():
                test_loss, test_acc = maml.test_on_task(fast_weight_list[i], test_x, test_y)
                losses.append(test_loss.detach().cpu().numpy())
                accs.append(test_acc.detach().cpu().numpy())

        preupdate_losses.append(preupdate_loss.detach().cpu().numpy())
        postupdate_losses.append(losses)
        postupdate_accs.append(accs)
    return postupdate_accs, postupdate_losses, preupdate_losses


def move_to_gpu(*args, device):
    return tuple([torch.tensor(x, dtype=torch.float).to(device) for x in arg] for arg in args)


if __name__ == '__main__':
    arguments = argparser.parse_args()

    print(arguments)

    main(arguments)
