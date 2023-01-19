# -*- coding: utf-8 -*-
# from torch._C import T
# from train import Trainer
import sys
sys.path.append('./src')

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from IPython import embed
import wandb
from ankge.utils import setup_parser
from ankge.utils.tools import *
from ankge.data.Sampler import *
from ankge.data.Grounding import GroundAllRules

def main():
    parser = setup_parser() 
    args = parser.parse_args()
    if args.load_config:
        args = load_config(args, args.config_path)
    seed_everything(args.seed) 
    """set up sampler to datapreprocess""" 

    if args.init_checkpoint:
        override_config(args) 
    elif args.data_path is None :
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args=args)

    logging.info("++++++++++++++++++++++++++loading hyper parameter++++++++++++++++++++++++++")
    for key, value in args.__dict__.items():
        logging.info("Parameter "+key+":  "+str(value))
    logging.info("++++++++++++++++++++++++++++++++over loading+++++++++++++++++++++++++++++++")


    train_sampler_class = import_class(f"ankge.data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  
    test_sampler_class = import_class(f"ankge.data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  
    """set up datamodule""" 
    data_class = import_class(f"ankge.data.{args.data_class}")
    kgdata = data_class(args, train_sampler, test_sampler)
    """set up model"""
    model_class = import_class(f"ankge.model.{args.model_name}")
    
    model = model_class(args)
    """set up lit_model"""
    litmodel_class = import_class(f"ankge.lit_model.{args.litmodel_name}")
    lit_model = litmodel_class(model, args)
    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="AnKGE")
        logger.log_hyperparams(vars(args))
    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval|mrr",
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )
    """set up model save method"""

    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval|mrr",
        mode="max",
        filename="{epoch}-{Eval|mrr:.3f}",
        dirpath=dirpath,
        save_weights_only=True,
        save_top_k=1,
    )
    callbacks = [early_callback, model_checkpoint]
    # initialize trainer

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus="0,",
        check_val_every_n_epoch=args.check_per_epoch,
    )
    '''保存参数到config'''
    if args.save_config:
        save_config(args)
    if args.use_wandb:
        logger.watch(lit_model)
    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        path = model_checkpoint.best_model_path
    else:
        path = args.checkpoint_dir
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)

if __name__ == "__main__":
    main()
