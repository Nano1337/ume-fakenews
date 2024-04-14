import torch 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

def run_trainer(args, model, train_loader, val_loader, test_loader, overfit_batches=0):
    """
    Run the pytorch lightning trainer with the given model and data loaders
    """
    
    # define trainer
    trainer = None
    if args.use_wandb:
        wandb_logger = WandbLogger(
            group=args.group_name,
        )
        wandb_run_id = wandb_logger.experiment.name

    # define callbacks
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    if args.use_wandb:
        file_name = wandb_run_id + "_best"
    else:
        file_name = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "_best"

    checkpoint_logger = pl.callbacks.ModelCheckpoint(
        dirpath=args.data_path + "_ckpts/" + args.group_name + "/",
        filename=file_name,
        save_top_k=1,
        monitor="val_epoch/val_avg_acc",
        mode="max",
    )

    # log config to wandb
    if args.use_wandb:
        wandb_logger.log_hyperparams(args)

    if torch.cuda.is_available():
        # call pytorch lightning trainer 
        trainer = pl.Trainer(
            strategy="auto",
            max_epochs=args.num_epochs, 
            logger = wandb_logger if args.use_wandb else None,
            deterministic=True, 
            default_root_dir="ckpts/",  
            precision="bf16-mixed", # "bf16-mixed",
            num_sanity_val_steps=0, # check validation 
            log_every_n_steps=30,  
            callbacks=[
                lr_logger, 
                checkpoint_logger,
                ],
            overfit_batches=overfit_batches, # use 1.0 to check if model is working
        )
    else: 
        raise NotImplementedError("It is not advised to train without a GPU")
    
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader, 
    )
    
    model.load_state_dict(torch.load(checkpoint_logger.best_model_path)["state_dict"]) 

    trainer.test(
        model, 
        dataloaders=test_loader,
    )


