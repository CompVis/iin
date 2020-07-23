import torch
import numpy as np
from edflow.util import retrieve, get_obj_from_str

from iin.iterators.base import Iterator
from iin.iterators.ae import sample_fid_callback


def loss_callback(root, data_in, data_out, config):
    log = {"scalars": dict()}
    log["scalars"]["loss"] = np.mean(data_out.labels["loss"])
    return log


class Trainer(Iterator):
    """
    Unsupervised IIN Trainer. Expects `image` in batch,
    `encode -> Distribution` and `decode` methods on first stage model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_stage_config = self.config["first_stage"]
        self.init_first_stage(first_stage_config)

    def init_first_stage(self, config):
        subconfig = config["subconfig"]
        self.first_stage = get_obj_from_str(config["model"])(subconfig)
        if "checkpoint" in config:
            checkpoint = config["checkpoint"]
            state = torch.load(checkpoint)["model"]
            self.first_stage.load_state_dict(state)
            self.logger.info("Restored first stage from {}".format(checkpoint))
        self.first_stage.to(self.device)
        self.first_stage.eval()

    def step_op(self, *args, **kwargs):
        inputs = kwargs["image"]
        inputs = self.totorch(inputs)

        with torch.no_grad():
            posterior = self.first_stage.encode(inputs)
            z = posterior.sample()
        z_ss, logdet = self.model(z)
        loss, log_dict, loss_train_op = self.loss(z_ss, logdet, self.get_global_step())
        log_dict.setdefault("scalars", dict())
        log_dict.setdefault("images", dict())

        def train_op():
            loss_train_op()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_lr()

        def log_op():
            if not hasattr(self, "fixed_examples"):
                self.fixed_examples = np.random.RandomState(1).randn(*posterior.mode().shape)
                self.fixed_examples = self.totorch(self.fixed_examples)

            with torch.no_grad():
                reconstructions = self.first_stage.decode(
                    self.model.reverse(z_ss))
                samples = self.first_stage.decode(
                    self.model.reverse(torch.randn_like(posterior.mode())))
                fixed_samples = self.first_stage.decode(
                    self.model.reverse(self.fixed_examples))

            log_dict["images"]["inputs"] = inputs
            log_dict["images"]["reconstructions"] = reconstructions
            log_dict["images"]["samples"] = samples
            log_dict["images"]["fixed_samples"] = fixed_samples

            for k in log_dict:
                for kk in log_dict[k]:
                    log_dict[k][kk] = self.tonp(log_dict[k][kk])

            for i, param_group in enumerate(self.optimizer.param_groups):
                log_dict["scalars"]["lr_{:02}".format(i)] = param_group['lr']
            return log_dict

        def eval_op():
            with torch.no_grad():
                loss_ = torch.ones(inputs.shape[0])*loss
                samples = self.first_stage.decode(self.model.reverse(torch.randn_like(posterior.mode())))
            return {"samples": self.tonp(samples),
                    "labels": {"loss": self.tonp(loss_)}}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        cbs = {"eval_op": {"loss_cb": loss_callback}}
        cbs["eval_op"]["fid_samples"] = sample_fid_callback
        return cbs


class FactorTrainer(Trainer):
    def step_op(self, *args, **kwargs):
        # get inputs
        inputs_factor = dict()
        z_factor = dict()
        z_ss_factor = dict()
        logdet_factor = dict()
        factors = kwargs["factor"]
        for k in ["example1", "example2"]:
            inputs = kwargs[k]["image"]
            inputs = self.totorch(inputs)
            inputs_factor[k] = inputs

            with torch.no_grad():
                posterior = self.first_stage.encode(inputs)
                z = posterior.sample()
                z_factor[k] = z
            z_ss, logdet = self.model(z)
            z_ss_factor[k] = z_ss
            logdet_factor[k] = logdet

        loss, log_dict, loss_train_op = self.loss(
            z_ss_factor, logdet_factor, factors, self.get_global_step())

        def train_op():
            loss_train_op()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_lr()

        def log_op():
            if not hasattr(self.model, "decode"):
                return None
            with torch.no_grad():
                for k in ["example1", "example2"]:
                    log_dict["images"][k] = inputs_factor[k]
                # reencode after update of model
                ss_z1 = self.model(z_factor["example1"])[0]
                ss_z2 = self.model(z_factor["example2"])[0]

                log_dict["images"]["reconstruction"] = self.first_stage.decode(
                        self.model.reverse(ss_z1))

                factor_mask = [
                        torch.tensor(((factors==i) | ((factors<0) & (factors!=-i)))[:,None,None,None]).to(
                            ss_z1[i]) for i in range(len(ss_z1))]
                ss_z_swap = [
                        ss_z1[i] +
                        factor_mask[i]*(
                            ss_z2[i] - ss_z1[i])
                        for i in range(len(factor_mask))]
                log_dict["images"]["decoded_swap"] = self.first_stage.decode(
                        self.model.reverse(ss_z_swap))

                N_cross = 6
                z_cross = z_factor["example1"][:N_cross,...]
                shape = tuple(z_cross.shape)
                z_cross1 = z_cross[None,...][N_cross*[0],...].reshape(
                    N_cross*N_cross, *shape[1:])
                z_cross2 = z_cross[:,None,...][:,N_cross*[0],...].reshape(
                    N_cross*N_cross, *shape[1:])
                ss_z_cross1 = self.model(z_cross1)[0]
                ss_z_cross2 = self.model(z_cross2)[0]
                for i in range(len(ss_z1)):
                    ss_z_cross = list(ss_z_cross2)
                    ss_z_cross[i] = ss_z_cross1[i]
                    log_dict["images"]["decoded_cross_{}".format(i)] = self.first_stage.decode(
                            self.model.reverse(ss_z_cross))

                N_fixed = 6
                if not hasattr(self, "fixed_examples"):
                    self.fixed_examples = [
                        self.dataset[i]["example1"]["image"]
                        for i in np.random.RandomState(1).choice(len(self.dataset),
                                                                 N_fixed)]
                    self.fixed_examples = np.stack(self.fixed_examples)
                    self.fixed_examples = self.totorch(self.fixed_examples)
                    self.fixed_examples = self.first_stage.encode(self.fixed_examples).mode()
                    shape = tuple(self.fixed_examples.shape)
                    self.fixed1 = self.fixed_examples[None,...][N_fixed*[0],...].reshape(
                        N_fixed*N_fixed, *shape[1:])
                    self.fixed2 = self.fixed_examples[:,None,...][:,N_fixed*[0],...].reshape(
                        N_fixed*N_fixed, *shape[1:])
                
                ss_z_fixed1 = self.model(self.fixed1)[0]
                ss_z_fixed2 = self.model(self.fixed2)[0]
                for i in range(len(ss_z_fixed1)):
                    ss_z_cross = list(ss_z_fixed2)
                    ss_z_cross[i] = ss_z_fixed1[i]
                    log_dict["images"]["fixed_cross_{}".format(i)] = self.first_stage.decode(
                            self.model.reverse(ss_z_cross))

                for k in log_dict:
                    for kk in log_dict[k]:
                        log_dict[k][kk] = self.tonp(log_dict[k][kk])
            return log_dict

        return {"train_op": train_op, "log_op": log_op}

    @property
    def callbacks(self):
        return dict()
