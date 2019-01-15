import torch
from sigver.featurelearning.data import extract_features
import sigver.featurelearning.models as models
import argparse
from sigver.datasets.util import load_dataset, get_subset
import sigver.wd.training as training
import numpy as np
import pickle


def main(args):
    exp_users = range(*args.exp_users)
    dev_users = range(*args.dev_users)

    assert len(
        set(exp_users).intersection(set(dev_users))) == 0, 'Exploitation set and Development set must not overlap'

    state_dict, class_weights, forg_weights = torch.load(args.model_path,
                                                                  map_location=lambda storage, loc: storage)
    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    base_model = models.available_models[args.model]().to(device).eval()

    base_model.load_state_dict(state_dict)

    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)

    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)

    features = extract_features(x, process_fn, args.batch_size, args.input_size)

    data = (features, y, yforg)

    exp_set = get_subset(data, exp_users)
    dev_set = get_subset(data, dev_users)

    rng = np.random.RandomState(1234)

    eer_u_list = []
    eer_list = []
    all_results = []
    for _ in range(args.folds):
        classifiers, results = training.train_test_all_users(exp_set,
                                                             dev_set,
                                                             svm_type=args.svm_type,
                                                             C=args.svm_c,
                                                             gamma=args.svm_gamma,
                                                             num_gen_train=args.gen_for_train,
                                                             num_forg_from_exp=args.forg_from_exp,
                                                             num_forg_from_dev=args.forg_from_dev,
                                                             num_gen_test=args.gen_for_test,
                                                             rng=rng)
        this_eer_u, this_eer = results['all_metrics']['EER_userthresholds'], results['all_metrics']['EER']
        all_results.append(results)
        eer_u_list.append(this_eer_u)
        eer_list.append(this_eer)
    print('EER (global threshold): {:.2f} (+- {:.2f})'.format(np.mean(eer_list) * 100, np.std(eer_list) * 100))
    print('EER (user thresholds): {:.2f} (+- {:.2f})'.format(np.mean(eer_u_list) * 100, np.std(eer_u_list) * 100))

    if args.save_path is not None:
        print('Saving results to {}'.format(args.save_path))
        with open(args.save_path, 'wb') as f:
            pickle.dump(all_results, f)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', choices=models.available_models, required=True,
                        help='Model architecture', dest='model')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--save-path')
    parser.add_argument('--input-size', nargs=2, default=(150, 220))

    parser.add_argument('--exp-users', type=int, nargs=2, default=(0, 300))
    parser.add_argument('--dev-users', type=int, nargs=2, default=(300, 881))

    parser.add_argument('--gen-for-train', type=int, default=12)
    parser.add_argument('--gen-for-test', type=int, default=10)
    parser.add_argument('--forg-from_exp', type=int, default=0)
    parser.add_argument('--forg-from_dev', type=int, default=14)

    parser.add_argument('--svm-type', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--svm-c', type=float, default=1)
    parser.add_argument('--svm-gamma', type=float, default=2**-11)

    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--folds', type=int, default=10)

    arguments = parser.parse_args()
    print(arguments)

    main(arguments)
