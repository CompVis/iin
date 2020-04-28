import torch
import numpy as np
from perceptual_similarity import PerceptualLoss
from edflow.util import retrieve
from fid import fid_callback

from iin.iterators.base import Iterator


def rec_fid_callback(*args, **kwargs):
    return fid_callback.fid(*args, **kwargs,
                            im_in_key="image",
                            im_in_support="-1->1",
                            im_out_key="reconstructions",
                            im_out_support="0->255",
                            name="fid_recons")


def sample_fid_callback(*args, **kwargs):
    return fid_callback.fid(*args, **kwargs,
                            im_in_key="image",
                            im_in_support="-1->1",
                            im_out_key="samples",
                            im_out_support="0->255",
                            name="fid_samples")


def reconstruction_callback(root, data_in, data_out, config):
    log = {"scalars": dict()}
    log["scalars"]["rec_loss"] = np.mean(data_out.labels["rec_loss"])
    log["scalars"]["kl_loss"] = np.mean(data_out.labels["kl_loss"])
    return log


class Trainer(Iterator):
    """
    AE Trainer. Expects `image` in batch, `encode -> Distribution` and `decode`
    methods on model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss = PerceptualLoss()

    def step_op(self, *args, **kwargs):
        inputs = kwargs["image"]
        inputs = self.totorch(inputs)

        posterior = self.model.encode(inputs)
        z = posterior.sample()
        reconstructions = self.model.decode(z)
        loss, log_dict, loss_train_op = self.loss(
            inputs, reconstructions, posterior, self.get_global_step())
        log_dict.setdefault("scalars", dict())
        log_dict.setdefault("images", dict())

        def train_op():
            loss_train_op()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_lr()

        def log_op():
            log_dict["images"]["inputs"] = inputs
            log_dict["images"]["reconstructions"] = reconstructions

            if not hasattr(self, "fixed_examples"):
                self.fixed_examples = [
                    self.dataset[i]["image"]
                    for i in np.random.RandomState(1).choice(len(self.dataset),
                                                             self.config["batch_size"])]
                self.fixed_examples = np.stack(self.fixed_examples)
                self.fixed_examples = self.totorch(self.fixed_examples)

            with torch.no_grad():
                log_dict["images"]["fixed_inputs"] = self.fixed_examples
                log_dict["images"]["fixed_reconstructions"] = self.model.decode(
                        self.model.encode(self.fixed_examples).sample())
                log_dict["images"]["decoded_sample"] = self.model.decode(
                        torch.randn_like(posterior.mode()))

            for k in log_dict:
                for kk in log_dict[k]:
                    log_dict[k][kk] = self.tonp(log_dict[k][kk])

            for i, param_group in enumerate(self.optimizer.param_groups):
                log_dict["scalars"]["lr_{:02}".format(i)] = param_group['lr']
            return log_dict

        def eval_op():
            with torch.no_grad():
                kl_loss  = posterior.kl()
                rec_loss = self.eval_loss(reconstructions, inputs)
                samples = self.model.decode(torch.randn_like(posterior.mode()))
            return {"reconstructions": self.tonp(reconstructions),
                    "samples": self.tonp(samples),
                    "labels": {"rec_loss": self.tonp(rec_loss),
                               "kl_loss": self.tonp(kl_loss)}}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        cbs = {"eval_op": {"reconstruction": reconstruction_callback}}
        cbs["eval_op"]["fid_reconstruction"] = rec_fid_callback
        cbs["eval_op"]["fid_samples"] = sample_fid_callback
        return cbs
