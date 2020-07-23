import albumentations
import numpy as np
from edflow.util import retrieve
from edflow.data.dataset import PRNGMixin, DatasetMixin, SubDataset
from edflow.datasets.mnist import (
    MNISTTrain as _MNISTTrain,
    MNISTTest as _MNISTTest)
from edflow.datasets.fashionmnist import (
    FashionMNISTTrain as _FashionMNISTTrain,
    FashionMNISTTest as _FashionMNISTTest)
from edflow.datasets.cifar import (
    CIFAR10Train as _CIFAR10Train,
    CIFAR10Test as _CIFAR10Test)
from edflow.datasets.celeba import (
    CelebATrain as _CelebATrain,
    CelebATest as _CelebATest)


class Base32(DatasetMixin, PRNGMixin):
    """Add support for resizing, cropping and dequantization."""
    def __init__(self, config):
        self.data = self.get_base_data(config)
        self.size = retrieve(config, "spatial_size", default=32)

        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def preprocess_example(self, example):
        example["image"] = ((example["image"]+1)*127.5).astype(np.uint8)
        example["image"] = self.preprocessor(image=example["image"])["image"]
        example["image"] = (example["image"] + self.prng.random())/256. # dequantization
        example["image"] = (example["image"]*2.0-1.0).astype(np.float32)
        return example

    def get_example(self, i):
        example = super().get_example(i)
        example = self.preprocess_example(example)
        return example


class MNISTTrain(Base32):
    def get_base_data(self, config):
        return _MNISTTrain(config)


class MNISTTest(Base32):
    def get_base_data(self, config):
        return _MNISTTest(config)


class FashionMNISTTrain(Base32):
    def get_base_data(self, config):
        return _FashionMNISTTrain(config)


class FashionMNISTTest(Base32):
    def get_base_data(self, config):
        return _FashionMNISTTest(config)


class CIFAR10Train(Base32):
    def get_base_data(self, config):
        return _CIFAR10Train(config)


class CIFAR10Test(Base32):
    def get_base_data(self, config):
        return _CIFAR10Test(config)


class BaseCelebA(DatasetMixin, PRNGMixin):
    """Add support for resizing, cropping and dequantization."""
    def __init__(self, config):
        self.data = self.get_base_data(config)
        self.size = retrieve(config, "spatial_size", default=64)
        self.attribute_descriptions = [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
            "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
            "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
            "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses",
            "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes",
            "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
            "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling",
            "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
            "Wearing_Necktie", "Young"]

        self.cropper = albumentations.CenterCrop(height=160,width=160)
        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        self.preprocessor = albumentations.Compose([self.cropper, self.rescaler])

    def preprocess_example(self, example):
        example["image"] = ((example["image"]+1)*127.5).astype(np.uint8)
        example["image"] = self.preprocessor(image=example["image"])["image"]
        example["image"] = (example["image"] + self.prng.random())/256. # dequantization
        example["image"] = (example["image"]*2.0-1.0).astype(np.float32)
        return example

    def get_example(self, i):
        example = super().get_example(i)
        example = self.preprocess_example(example)
        return example


class CelebATrain(BaseCelebA):
    def get_base_data(self, config):
        return _CelebATrain(config)


class CelebATest(BaseCelebA):
    def get_base_data(self, config):
        data = _CelebATest(config)
        indices = np.random.RandomState(1).choice(len(data), size=10000)
        return SubDataset(data, indices)


class BaseFactorCelebA(BaseCelebA):
    def __init__(self, config):
        super().__init__(config)
        self.attributes = retrieve(config, "CelebAFactors/attributes",
                                   default=["Eyeglasses", "Male", "No_Beard", "Smiling"])
        self.attribute_indices = [self.attribute_descriptions.index(attr)
                for attr in self.attributes]
        self.pos_attribute_choices = [np.where(self.labels["attributes"][:,attridx]==1)[0]
                for attridx in self.attribute_indices]
        self.neg_attribute_choices = [np.where(self.labels["attributes"][:,attridx]==-1)[0]
                for attridx in self.attribute_indices]
        self.n_factors = len(self.attributes)+1
        self.residual_index = len(self.attributes)

    def get_factor_idx(self, i):
        factor = self.prng.choice(len(self.attributes))
        attridx = self.attribute_indices[factor]
        attr = self.labels["attributes"][i,attridx]
        if attr == 1:
            i2 = self.prng.choice(self.pos_attribute_choices[factor])
        else:
            i2 = self.prng.choice(self.neg_attribute_choices[factor])
        return factor, i2

    def get_example(self, i):
        e1 = super().get_example(i)
        factor, i2 = self.get_factor_idx(i)
        e2 = super().get_example(i2)
        example = {
                "factor": factor,
                "example1": e1,
                "example2": e2}
        return example


class FactorCelebATrain(BaseFactorCelebA):
    def get_base_data(self, config):
        return _CelebATrain(config)


class FactorCelebATest(BaseFactorCelebA):
    def __init__(self, config):
        super().__init__(config)
        self.test_prng = np.random.RandomState(1)
        self.factor_idx = [BaseFactorCelebA.get_factor_idx(self, i) for i in range(len(self))]

    @property
    def prng(self):
        return self.test_prng

    def get_factor_idx(self, i):
        return self.factor_idx[i]

    def get_base_data(self, config):
        data = _CelebATest(config)
        indices = np.random.RandomState(1).choice(len(data), size=10000)
        return SubDataset(data, indices)


class ColorfulMNISTBase(DatasetMixin):
    def get_factor(self, i):
        factor = self.prng.choice(2)
        return factor

    def get_same_idx(self, i):
        cls = self.labels["class"][i]
        others = np.where(self.labels["class"] == cls)[0]
        return self.prng.choice(others)

    def get_other_idx(self, i):
        return self.prng.choice(len(self))

    def get_color(self, i):
        return self.prng.uniform(low=0,high=1,size=3).astype(np.float32)

    def get_example(self, i):
        example1 = super().get_example(i)
        factor = self.get_factor(i)
        example = {"factor": factor, "example1": example1}

        if factor == 0:
            # same digit, different color
            j = self.get_same_idx(i)
            color1 = self.get_color(i)
            color2 = self.get_color(j)
        else:
            # different digit, same color
            j = self.get_other_idx(i)
            color1 = self.get_color(i)
            color2 = color1

        example2 = super().get_example(j)
        example["example2"] = example2

        example["example1"]["image"] = example["example1"]["image"] * color1
        example["example2"]["image"] = example["example2"]["image"] * color2

        return example


class ColorfulMNISTTrain(ColorfulMNISTBase, PRNGMixin):
    def __init__(self, config):
        self.data = MNISTTrain(config)
        self.n_factors = 3
        self.residual_index = 2


class ColorfulMNISTTest(ColorfulMNISTBase):
    def __init__(self, config):
        self.data = MNISTTest(config)
        self.prng = np.random.RandomState(1)
        self.factor = [ColorfulMNISTBase.get_factor(self, i) for i in range(len(self))]
        self.same_idx = [ColorfulMNISTBase.get_same_idx(self, i) for i in range(len(self))]
        self.other_idx = [ColorfulMNISTBase.get_other_idx(self, i) for i in range(len(self))]
        self.color = [ColorfulMNISTBase.get_color(self, i) for i in range(len(self))]
        self.n_factors = 3
        self.residual_index = 2

    def get_factor(self, i):
        return self.factor[i]

    def get_same_idx(self, i):
        return self.same_idx[i]

    def get_other_idx(self, i):
        return self.other_idx[i]

    def get_color(self, i):
        return self.color[i]


class SingleColorfulMNISTTrain(DatasetMixin):
    def __init__(self, config):
        self.data = ColorfulMNISTTrain(config)

    def get_example(self, i):
        return super().get_example(i)["example1"]


class SingleColorfulMNISTTest(DatasetMixin):
    def __init__(self, config):
        self.data = ColorfulMNISTTest(config)

    def get_example(self, i):
        return super().get_example(i)["example1"]
