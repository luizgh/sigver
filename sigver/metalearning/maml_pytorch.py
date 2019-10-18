import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_value_
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


class noop_callback:
    def scalar(*args, **kwargs):
        pass


class MAML:
    def __init__(self, model, num_updates, test_num_updates, task_lr, meta_task_lr,
                 meta_task_min_lr, max_epochs, learn_task_lr, weights, device, callback=None,
                 loss_function=F.cross_entropy, is_classification=True):
        self.model = model
        self.num_updates = num_updates
        self.test_num_updates = test_num_updates
        self.task_lr = torch.full((num_updates, len(weights)), task_lr,
                                  requires_grad=True, device=device)

        self.meta_task_lr = meta_task_lr
        self.callback = callback
        self.loss_function = loss_function
        self.is_classification = is_classification
        self.weights = weights

        model_params = list(model.parameters())

        self.parameters = model_params + list(weights.values())

        if learn_task_lr:
            self.parameters.append(self.task_lr)

        self.optimizer = Adam(self.parameters, meta_task_lr)
        self.scheduler = CosineAnnealingLR(
            optimizer=self.optimizer, T_max=max_epochs,
            eta_min=meta_task_min_lr)

        self.callback = callback or noop_callback()

    def load(self, param_list):
        for param, param_value in zip(self.parameters, param_list):
            param.data.copy_(param_value)

    def meta_learning_step(self, trainset, testset, loss_weights, step):
        x_train, y_train = trainset
        x_test, y_test = testset

        preupdate_losses = []
        postupdate_losses = []
        postupdate_accs = []

        # First dimension of trainset is the episodes. Iterate one at a time,
        # accumulating the gradients.

        meta_batch_size = len(x_train)
        self.optimizer.zero_grad()
        
        for n_batch in range(meta_batch_size):
            fast_weight_list = self.adapt_weights_to_task(x_train[n_batch], y_train[n_batch], self.num_updates)

            # Monitor performance on each num_updates
            preupdate_loss, preupdate_acc = self.test_on_task(self.weights, x_test[n_batch], y_test[n_batch])
            all_losses = []
            all_accs = []
            loss = 0
            for n_update in range(self.num_updates):
                test_loss, test_acc = self.test_on_task(fast_weight_list[n_update], x_test[n_batch], y_test[n_batch])
                all_losses.append(test_loss.detach().cpu().numpy())
                all_accs.append(test_acc.detach().cpu().numpy())

                loss += test_loss * loss_weights[n_update]

            # test_loss.backward()  # will get the test_loss from the last iteration
            loss.backward()
            preupdate_losses.append(preupdate_loss.detach().cpu().numpy())
            postupdate_losses.append(all_losses)
            postupdate_accs.append(all_accs)

        # Apply the gradients. Use gradient cliping first.
        self.normalize_gradients(meta_batch_size)
        grad = {k: v.grad for k, v in self.weights.items()}
        clip_grad_value_(self.parameters, 10)
        self.optimizer.step()

        # Record statistics
        # self.callback.scalar('preupdate_loss', step, np.mean(preupdate_losses))
        for n_update in range(self.num_updates):
            self.callback.scalar('postupdate_loss_{}'.format(n_update),
                                 step, np.mean(postupdate_losses, axis=0)[n_update])

            if self.is_classification:
                self.callback.scalar('postupdate_acc_{}'.format(n_update),
                                     step, np.mean(postupdate_accs, axis=0)[n_update])

        return preupdate_losses, postupdate_losses, postupdate_accs, grad

    def test_on_task(self, fast_weights, x_test, y_test):
        test_logits = self.model.forward(x_test, fast_weights, training=False)
        test_loss = self.loss_function(test_logits, y_test)

        if self.is_classification:
            if test_logits.shape[1] == 1:  # binary classification
                test_pred = test_logits.detach() > 0
            else:
                test_pred = test_logits.detach().argmax(1)
            test_acc = y_test.type(torch.uint8).eq(test_pred).float().mean()
        else:
            test_acc = None

        return test_loss, test_acc

    def adapt_weights_to_task(self, x_train, y_train, num_updates, is_training=True):
        logits = self.model.forward(x_train, self.weights, training=is_training)
        preupdate_loss = self.loss_function(logits, y_train)

        grad = torch.autograd.grad(preupdate_loss, list(self.weights.values()), create_graph=True)

        fast_weights = {name: old_value - lr * g
                        for (name, old_value), g, lr in zip(self.weights.items(), grad, self.task_lr[0])}

        fast_weight_list = [fast_weights]

        for i in range(num_updates - 1):
            logits = self.model.forward(x_train, fast_weights, training=is_training)
            loss = self.loss_function(logits, y_train)

            grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            fast_weights = {name: old_value - lr * g
                            for (name, old_value), g, lr in zip(fast_weights.items(), grad, self.task_lr[i+1])}

            fast_weight_list.append(fast_weights)

        # If we want to Update BN running mean/std (e.g. with momentum=1), we can use this:
        # self.model.forward(x_train, fast_weights, training=True)

        return fast_weight_list

    def normalize_gradients(self, meta_batch_size):
        for p in filter(lambda p: p.grad is not None, self.parameters):
            p.grad.data.div_(meta_batch_size)



def get_per_step_loss_importance_vector(number_of_training_steps_per_iter,
                                        multi_step_loss_num_epochs,
                                        current_epoch):
    """
    adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch

    Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
    loss towards the optimization loss.
    :return: A tensor to be used to compute the weighted average of the loss, useful for
    the MSL (Multi Step Loss) mechanism.
    """

    if multi_step_loss_num_epochs == 0:
        loss_weights = torch.zeros(number_of_training_steps_per_iter)
        loss_weights[-1] = 1
        return loss_weights

    loss_weights = np.ones(shape=(number_of_training_steps_per_iter)) * (
            1.0 / number_of_training_steps_per_iter)
    decay_rate = 1.0 / number_of_training_steps_per_iter / multi_step_loss_num_epochs

    min_value_for_non_final_losses = 0.03 / number_of_training_steps_per_iter
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (current_epoch * (number_of_training_steps_per_iter - 1) * decay_rate),
        1.0 - ((number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights)
    return loss_weights


def balanced_binary_cross_entropy(logits, y):
    positive_count = y.sum()
    negative_count = len(y) - positive_count

    if positive_count != 0 and negative_count != 0:
        weights = y / (2 * positive_count) + (1 - y) / (2 * negative_count)
        ce = F.binary_cross_entropy_with_logits(logits, y, weight=weights, reduction='sum')
    else:
        ce = F.binary_cross_entropy_with_logits(logits, y)
    return ce
