import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from .network import create_model, virtual_layer

BATCH_SIZE = 100

class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()
        self.fc = virtual_layer(num_classes=2)
        
    def forward(self, x, y):
        out = self.model(x)
        out = self.fc(out, y)
        
        return F.log_softmax(out, dim=1)
    
    def get_embeddings(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x, y)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss


    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x, y)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")


    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=1e-3,)
        
        steps_per_epoch = 45000 // BATCH_SIZE
        
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.hparams.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    