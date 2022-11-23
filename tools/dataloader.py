from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision

class CifarDataModule(pl.LightningDataModule):
    def __init__(self, data_dir = 'data' , batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = torchvision.transforms.Compose(
                                [
                                    torchvision.transforms.RandomCrop(32, padding=4),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    cifar10_normalization(),
                                ]
                            )
        self.transform_test  = torchvision.transforms.Compose(
                                [
                                    torchvision.transforms.ToTensor(),
                                    cifar10_normalization(),
                                ]
                            )
        self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform_test)
            
    #Download data
    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
    
    # Create train/val split
    def setup(self, stage=None):
        print(stage)
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
            
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True,  num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cifar_val,   batch_size=self.batch_size, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.cifar_test,  batch_size=100, shuffle=False, num_workers=8)