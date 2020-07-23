import torch
import numpy as np
from edflow.util import retrieve

from iin.iterators.base import Iterator


class Trainer(Iterator):
    """
    Classification Trainer. Expects `image` and `class` in batch,
    and a model returning logits.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = retrieve(self.config, "n_classes")

    def step_op(self, *args, **kwargs):
        inputs = kwargs["image"]
        inputs = self.totorch(inputs)
        labels = kwargs["class"]
        labels = self.totorch(labels).to(torch.int64)
        onehot = torch.nn.functional.one_hot(labels,
                                             num_classes=self.n_classes).float()

        logits = self.model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,
                                                                    onehot,
                                                                    reduction="none")
        mean_loss = loss.mean()

        def train_op():
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()
            self.update_lr()

        def log_op():
            with torch.no_grad():
                prediction = torch.argmax(logits, dim=1)
                accuracy = (prediction==labels).float().mean()

                log_dict = {"scalars": {
                    "loss": mean_loss,
                    "acc": accuracy
                }}

                for k in log_dict:
                    for kk in log_dict[k]:
                        log_dict[k][kk] = self.tonp(log_dict[k][kk])

                for i, param_group in enumerate(self.optimizer.param_groups):
                    log_dict["scalars"]["lr_{:02}".format(i)] = param_group['lr']
                return log_dict

        def eval_op():
            return {
                "labels": {"loss": self.tonp(loss), "logits": self.tonp(logits)},
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        return {"eval_op": {"clf_callback": clf_callback}}


def clf_callback(root, data_in, data_out, config):
    loss = data_out.labels["loss"].mean()
    prediction = data_out.labels["logits"].argmax(1)
    accuracy = (prediction == data_in.labels["class"][:prediction.shape[0]]).mean()
    return {"scalars": {"loss": loss, "accuracy": accuracy}}
