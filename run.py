import os
import torch
from pytorch_lightning import seed_everything
from tools import LitResnet, CifarDataModule
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import tensorflow as tf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

seed_everything(7)
PATH_DATASETS = os.environ.get("PATH_DATASETS", "data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

cifar10_dm = CifarDataModule()
model = LitResnet(lr=0.1)

trainer = Trainer(
    max_epochs=50,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=1)],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

model = model.load_from_checkpoint(f"logs/lightning_logs/version_21/checkpoints/epoch=0-step=176.ckpt")

embeddings = np.empty([0, 512])
labels = np.empty([0])
images = np.empty([0, 3, 32, 32])
for i, (x, y) in enumerate(tqdm(cifar10_dm.test_dataloader())):
    out = model.get_embeddings(x).detach().numpy()
    embeddings = np.append(embeddings, out, axis=0)
    images = np.append(images, x, axis=0)
    labels = np.append(labels, y, axis=0)
    
    if i > 2:
        break

writer = SummaryWriter()
writer.add_embedding(embeddings,
                     metadata=labels,
                     label_img=torch.tensor(images))