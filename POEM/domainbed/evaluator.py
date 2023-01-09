import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def accuracy_from_loader(algorithm, loader, weights, bool_angle=False, bool_task=False, env_num = -1, debug=False):
    correct = 0
    domain_correct = 0
    total = 0
    losssum = 0.0
    loss_task_sum = 0.0
    loss_angle_sum = 0.0
    domain_losssum = 0.0
    weights_offset = 0
    domain_label = None

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        B = len(x)

        if bool_angle or bool_task:
            if env_num == -1:
                continue
            d = [env_num] * int(x.shape[0])
            domain_label = torch.tensor(d).to(device)
            with torch.no_grad():
                domain_logits = algorithm.predict_domain(x)
                domain_loss = F.cross_entropy(domain_logits, domain_label).item()

            domain_losssum += domain_loss * B

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()


        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if bool_angle or bool_task:
            if domain_logits.size(1) == 1:
                domain_correct += (domain_logits.gt(0).eq(domain_label).float() * batch_weights).sum().item()
            else:
                domain_correct += (domain_logits.argmax(1).eq(domain_label).float() * batch_weights).sum().item()

        if bool_task:
            loss_task1 = F.cross_entropy(algorithm.predict_task(algorithm.extract(x)), torch.tensor([0] * int(x.shape[0])).to(device))
            loss_task2 = F.cross_entropy(algorithm.predict_task(algorithm.extract_domain(x)), torch.tensor([1] * int(x.shape[0])).to(device))
            loss_task_sum += (loss_task1.item() + loss_task2.item()) / (2 * total)

        if bool_angle:
            loss_angle_sum += torch.abs(F.cosine_similarity(algorithm.predict_task(algorithm.extract(x)), algorithm.extract_domain(x)), dim=1).item() / total

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total

    if bool_angle or bool_task:
        domain_acc = domain_correct / total
        domain_loss = domain_losssum / total

        return acc, loss, domain_acc, domain_loss, loss_task_sum, loss_angle_sum
    else:
        return acc, loss


def accuracy(algorithm, loader_kwargs, weights, bool_angle=False, bool_task=False, env_num=-1, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, bool_angle, bool_task, env_num, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None, bool_angle=False, bool_task=False
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.bool_angle = bool_angle
        self.bool_task = bool_task

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        summaries["domain_eva"] = 0.0
        summaries["task_eval_loss"] = 0.0
        summaries["angle_eval_loss"] = 0.0
        summaries["domain_eval_loss"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm, loader_kwargs, weights, debug=self.debug)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
                    if self.bool_angle or self.bool_task:
                        if env_num > self.test_envs[0]:
                            num = env_num - 1
                        else:
                            num = env_num

                        _, _, domain_acc, domain_loss, task_loss, angle_loss = accuracy(algorithm, loader_kwargs, weights,
                                                                 self.bool_angle, self.bool_task, num,
                                                                 debug=self.debug)
                        summaries["task_eval_loss"] += task_loss / n_train_envs
                        summaries["angle_eval_loss"] += angle_loss / n_train_envs
                        summaries["domain_eval_loss"] += domain_loss / n_train_envs
                        # accuracies["ac_d_" + name +"_"+str(num)] = domain_acc
                        # losses["lo_d_" + name+"_"+str(num)] = domain_loss
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs


        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
