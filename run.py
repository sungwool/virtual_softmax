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
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

seed_everything(7)
PATH_DATASETS = os.environ.get("PATH_DATASETS", "data")
BATCH_SIZE = 100
NUM_WORKERS = int(os.cpu_count() / 2)

data_module = MNISTDataModule() # CifarDataModule()
model = LitResnet(lr=1e-1)

checkpoint_callback = ModelCheckpoint(dirpath='logs', save_top_k =1, mode='max', monitor='val_acc')

trainer = Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[LearningRateMonitor(logging_interval="epoch"), TQDMProgressBar(refresh_rate=1), checkpoint_callback],
)

trainer.fit(model, data_module)

# model.load_state_dict(torch.load('logs/softmax_09.ckpt')['state_dict'])

embeddings = np.empty([0, 2])
labels = np.empty([0])
images = np.empty([0, 1, 28, 28])

for i, (x, y) in enumerate(tqdm(data_module.test_dataloader())):    
    out = model.get_embeddings(x).detach().numpy()
    embeddings = np.append(embeddings, out, axis=0)
    images = np.append(images, x, axis=0)
    labels = np.append(labels, y, axis=0)
    
pad = np.zeros((embeddings.shape[0], 1))
embeddings = np.concatenate((embeddings, pad), axis=-1)
    
writer = SummaryWriter()
writer.add_embedding(embeddings[:100],
                     metadata=labels[:100],
                     label_img=torch.tensor(images[:100])
                     )

idx = labels == 0
embeddings_ = embeddings[idx]
embeddings_norm_ = np.linalg.norm(embeddings_, 2, axis=1)
embeddings_idx = np.argsort(embeddings_norm_)
sorted_img = images[idx][embeddings_idx]
sorted_norm = embeddings_norm_[embeddings_idx]

sorted_img = np.squeeze(sorted_img, 1)
sorted_img = [Image.fromarray((sorted_img[i]*255).astype('uint8'), 'L') for i in range(len(sorted_img))]

fig = plt.figure(figsize=(100, 10))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(10, 100),
                 axes_pad=0.1,
                 )
for i, (ax, im) in enumerate(zip(grid, sorted_img)):
    ax.imshow(im)
    ax.axis('off')    
plt.savefig('results/virtual_49_4.png')

idx = labels == 1
embeddings_ = embeddings[idx]
embeddings_norm_ = np.linalg.norm(embeddings_, 1, axis=1)
embeddings_idx = np.argsort(embeddings_norm_)
sorted_img = images[idx][embeddings_idx]
sorted_norm = embeddings_norm_[embeddings_idx]

sorted_img = np.squeeze(sorted_img, 1)
sorted_img = [Image.fromarray((sorted_img[i]*255).astype('uint8'), 'L') for i in range(len(sorted_img))]

fig = plt.figure(figsize=(100, 10))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(10, 100),
                 axes_pad=0.1,
                 )
for i, (ax, im) in enumerate(zip(grid, sorted_img)):
    ax.imshow(im)
    ax.axis('off')    
plt.savefig('results/virtual_49_9.png')