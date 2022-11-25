import os
import torch
from pytorch_lightning import seed_everything
from tools import LitResnet, CifarDataModule, MNISTDataModule
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
from sklearn.decomposition import PCA
from pytorch_lightning.callbacks import ModelCheckpoint

seed_everything(7)
PATH_DATASETS = os.environ.get("PATH_DATASETS", "data")
BATCH_SIZE = 100
NUM_WORKERS = int(os.cpu_count() / 2)

data_module = MNISTDataModule() # CifarDataModule()
model = LitResnet(lr=1e-1)

checkpoint_callback = ModelCheckpoint(dirpath='logs', save_top_k =1, mode='max', monitor='val_acc')

trainer = Trainer(
    max_epochs=20,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="epoch"), TQDMProgressBar(refresh_rate=1), checkpoint_callback],
)

trainer.fit(model, data_module)

# model.load_state_dict(torch.load('logs/epoch=17-step=1422.ckpt')['state_dict'])

embeddings = np.empty([0, 2])
labels = np.empty([0])
images = np.empty([0, 1, 28, 28])

for i, (x, y) in enumerate(tqdm(data_module.test_dataloader())):
    out = model.get_embeddings(x).detach().numpy()
    print(out.shape)
    embeddings = np.append(embeddings, out, axis=0)
    images = np.append(images, x, axis=0)
    labels = np.append(labels, y, axis=0)
    
    if i > 2:
        pad = np.zeros((embeddings.shape[0], 1))
        print(pad.shape, embeddings.shape)
        embeddings = np.concatenate((embeddings, pad), axis=-1)
        break


writer = SummaryWriter()
writer.add_embedding(embeddings,
                     metadata=labels,
                     label_img=torch.tensor(images)
                     )