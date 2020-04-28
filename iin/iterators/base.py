import itertools
import torch
import numpy as np
from edflow import TemplateIterator, get_obj_from_str
from edflow.util import retrieve


def totorch(x, guess_image=True, device=None):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    x = torch.tensor(x)
    if guess_image and len(x.size()) == 4:
        x = x.transpose(2, 3).transpose(1, 2)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    x = x.to(device)
    return x


def tonp(x, guess_image=True):
    try:
        if guess_image and len(x.shape) == 4:
            x = x.transpose(1, 2).transpose(2, 3)
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def get_learning_rate(config):
    if "learning_rate" in config:
        learning_rate = config["learning_rate"]
    elif "base_learning_rate" in config:
        learning_rate = config["base_learning_rate"]*config["batch_size"]
    else:
        raise KeyError()
    return learning_rate


class Iterator(TemplateIterator):
    """
    Base class to handle device and state. Adds optimizer and loss.
    Call update_lr() in train op for lr scheduling.

    Config parameters:
        - test_mode : boolean : Put model into .eval() mode.
        - no_restore_keys : string1,string2 : Submodels which should not be
                                              restored from checkpoint.
        - learning_rate : float : Learning rate of Adam
        - base_learning_rate : float : Learning_rate per example to adjust for
                                       batch size (ignored if learning_rate is present)
        - decay_start : float : Step after which learning rate is decayed to
                                zero.
        - loss : string : Import path of loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.test_mode = self.config.get("test_mode", False)
        if self.test_mode:
            self.model.eval()
        self.submodules = ["model"]
        self.do_not_restore_keys = retrieve(self.config, 'no_restore_keys', default='').split(',')

        self.learning_rate = get_learning_rate(self.config)
        self.logger.info("learning_rate: {}".format(self.learning_rate))
        if "loss" in self.config:
            self.loss = get_obj_from_str(self.config["loss"])(self.config)
            self.loss.to(self.device)
            self.submodules.append("loss")
            self.optimizer = torch.optim.Adam(
                    itertools.chain(self.model.parameters(), self.loss.parameters()),
                    lr=self.learning_rate,
                    betas=(0.5, 0.9))
            self.submodules.append("optimizer")
            try:
                self.loss.set_last_layer(self.model.get_last_layer())
            except Exception as e:
                self.logger.info(' Could not set last layer for calibration. Reason:\n {}'.format(e))

        self.num_steps = retrieve(self.config, "num_steps")
        self.decay_start = retrieve(self.config, "decay_start", default=self.num_steps)

    def get_decay_factor(self):
        alpha = 1.0
        if self.num_steps > self.decay_start:
            alpha = 1.0 - np.clip(
                (self.get_global_step() - self.decay_start) /
                (self.num_steps - self.decay_start),
                0.0, 1.0)
        return alpha

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_decay_factor()*self.learning_rate

    def get_state(self):
        state = dict()
        for k in self.submodules:
            state[k] = getattr(self, k).state_dict()
        return state

    def save(self, checkpoint_path):
        torch.save(self.get_state(), checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        keys = list(state.keys())
        for k in keys:
            if hasattr(self, k):
                if k not in self.do_not_restore_keys:
                    try:
                        missing, unexpected = getattr(self, k).load_state_dict(state[k], strict=False)
                        if missing:
                            self.logger.info("Missing keys for {}: {}".format(k, missing))
                        if unexpected:
                            self.logger.info("Unexpected keys for {}: {}".format(k, unexpected))
                    except TypeError:
                        self.logger.info(k)
                        try:
                            getattr(self, k).load_state_dict(state[k])
                        except ValueError:
                            self.logger.info("Could not load state dict for key {}".format(k))
                    else:
                        self.logger.info('Restored key `{}`'.format(k))
                else:
                    self.logger.info('Not restoring key `{}` (as specified)'.format(k))
            del state[k]

    def totorch(self, x, guess_image=True):
        return totorch(x, guess_image=guess_image, device=self.device)

    def tonp(self, x, guess_image=True):
        return tonp(x, guess_image=guess_image)

    def interpolate_corners(self, x, num_side, permute=False):
        return interpolate_corners(x, side=num_side, permute=permute)
