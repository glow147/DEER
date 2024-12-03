import os
import torch
import random
import argparse
import numpy as np
import lightning as L

from omegaconf import OmegaConf

from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from models import DEER  
from datasets import OCRDataModule



def make_args_parser():
    parser = argparse.ArgumentParser("DEER : Detection-agnostic End-to-End Recognizer for Scene Text Spotting", add_help=False)
    parser.add_argument("--config",default="config.path",type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()
    return args

def main(config):
    data_module = OCRDataModule(config)
    model = DEER(config)

    if not config.test:
        data_module.setup(stage="fit")

        last_epoch_checkpoint = ModelCheckpoint(
            dirpath='./checkpoints/',
            filename='model-last-adam-{epoch:03d}',
            save_on_train_epoch_end=True,
            every_n_epochs=1,
            save_last=True 
        )

        best_f1_checkpoint = ModelCheckpoint(
            dirpath='./checkpoints/',
            filename='model-best-f1-adam-{epoch:03d}-{val_f1:.4f}',
            monitor='val_f1',
            mode='max', 
            save_top_k=3,
        )

        logger = WandbLogger(name="DEER",project="DEER")

        trainer = L.Trainer(
            max_steps=config.MODEL.PARAMS.MAX_STEPS * (32 // (config.MODEL.PARAMS.BATCH_SIZE * torch.cuda.device_count())),
            accelerator="auto",            
            enable_progress_bar=True, 
            strategy=DDPStrategy(),
            devices=torch.cuda.device_count(),
            check_val_every_n_epoch=10,
            gradient_clip_val=0.5,
            num_sanity_val_steps=1,
            gradient_clip_algorithm="norm",
            callbacks=[last_epoch_checkpoint, best_f1_checkpoint],
            logger=logger,
        )

        if config.MODEL.WEIGHTS:
            checkpoint = torch.load(config.MODEL.WEIGHTS, map_location=torch.device('cpu'))
            if "optimizer_states" in checkpoint:
                if config.finetune:
                    model.load_state_dict(checkpoint["state_dict"])
                    trainer.fit(model, datamodule=data_module)
                else:
                    trainer.fit(model, datamodule=data_module, ckpt_path=config.MODEL.WEIGHTS)
            else:
                model.load_state_dict(checkpoint)
                trainer.fit(model, datamodule=data_module)
        else:
            trainer.fit(model, datamodule=data_module)

    else:
        data_module.setup(stage="test")

        trainer = L.Trainer(
            accelerator="auto",
            devices=torch.cuda.device_count(),
            enable_progress_bar=True
        )
        checkpoint = torch.load(config.MODEL.WEIGHTS, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'] if 'optimizer_states' in checkpoint else checkpoint)

        trainer.test(model, data_module.test_dataloader())
    
if __name__ == "__main__":
    args = make_args_parser()
    config = OmegaConf.load(args.config)
    config.MODEL.WEIGHTS = args.weights
    config.test = args.test
    config.finetune = args.finetune

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main(config)
