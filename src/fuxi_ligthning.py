import pytorch_lightning as L

from src.fuxi import FuXi as FuXiBase
from src.score_torch import *


class FuXi(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model: FuXiBase = FuXiBase(
            25,
            config.get('model_parameter')['channel'],
            config.get('model_parameter')['transformer_blocks'],
            121, 240,
            heads=config.get('model_parameter')['heads'],
        )
        self.lr = config.get("learning_rate")

    def training_step(self, batch, batch_idx):
        ts, autoregression_steps, lat_weights = batch
        loss = self.model.step(ts, lat_weights, autoregression_steps=autoregression_steps)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ts, autoregression_steps, lat_weights = batch
        loss, outs = self.model.step(ts, lat_weights, autoregression_steps=autoregression_steps, return_out=True)

        rmse = compute_weighted_rmse(outs, ts[:, 2:, :, :, :].cpu(), lat_weights)
        acc = compute_weighted_acc(outs, ts[:, 2:, :, :, :].cpu(), lat_weights)
        mae = compute_weighted_mae(outs, ts[:, 2:, :, :, :].cpu(), lat_weights)
        self.log('val_loss', loss)
        self.log('val_rmse', rmse.mean())
        self.log('val_acc', acc.mean())
        self.log('val_mae', mae.mean())

        return {
            "loss": loss,
            "rmse": rmse,
            "acc": acc,
            "mae": mae
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        return optimizer
