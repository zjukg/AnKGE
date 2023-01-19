# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import pytorch_lightning as pl
from ankge import lit_model
from ankge import data
def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument('--model_name', default="TransE", type=str, help='The name of model.')
    parser.add_argument('--dataset_name', default="FB15K237", type=str, help='The name of dataset.')
    parser.add_argument('--data_class', default="KGDataModule", type=str, help='The name of data preprocessing module, default KGDataModule.')
    parser.add_argument("--litmodel_name", default="KGELitModel", type=str, help='The name of processing module of training, evaluation and testing, default KGELitModel.')
    parser.add_argument("--train_sampler_class",default="UniSampler",type=str, help='Sampling method used in training, default UniSampler.')
    parser.add_argument("--test_sampler_class",default="TestSampler",type=str, help='Sampling method used in validation and testing, default TestSampler.')
    parser.add_argument('--loss_name', default="Adv_Loss", type=str, help='The name of loss function.')
    parser.add_argument('--negative_adversarial_sampling','-adv', default=True, action='store_false', help='Use self-adversarial negative sampling.')
    parser.add_argument('--optim_name', default="Adam", type=str, help='The name of optimizer')
    parser.add_argument("--seed", default=321, type=int, help='Random seed.')
    parser.add_argument('--margin', default=12.0, type=float, help='The fixed margin in loss function. ')
    parser.add_argument('--adv_temp', default=1.0, type=float, help='The temperature of sampling in self-adversarial negative sampling.')
    parser.add_argument('--emb_dim', default=200, type=int, help='The embedding dimension in KGE model.')
    parser.add_argument('--out_dim', default=200, type=int, help='The output embedding dimmension in some KGE model.')
    parser.add_argument('--num_neg', default=10, type=int, help='The number of negative samples corresponding to each positive sample')
    parser.add_argument('--num_ent', default=None, type=int, help='The number of entity, autogenerate.')
    parser.add_argument('--num_rel', default=None, type=int, help='The number of relation, autogenerate.')
    parser.add_argument('--check_per_epoch', default=5, type=int, help='Evaluation per n epoch of training.')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='If the number of consecutive bad results is n, early stop.')
    parser.add_argument("--num_layers", default=2, type=int, help='The number of layers in some GNN model.')
    parser.add_argument('--regularization', '-r', default=0.0, type=float)
    parser.add_argument("--decoder_model", default=None, type=str, help='The name of decoder model, in some model.')
    parser.add_argument('--eval_task', default="link_prediction", type=str, help='The task of validation, default link_prediction.')
    parser.add_argument("--calc_hits", default=[1,3,10], type=lambda s: [int(item) for item in s.split(',')], help='calc hits list')
    parser.add_argument('--filter_flag', default=True, action='store_false', help='Filter in negative sampling.')
    parser.add_argument('--gpu', default='cuda:0', type=str, help='Select the GPU in training, default cuda:0.')
    parser.add_argument("--use_wandb", default=False, action='store_true',help='Use "weight and bias" to record the result.')
    parser.add_argument('--use_weight', default=False, action='store_true', help='Use subsampling weight.')
    parser.add_argument('--checkpoint_dir', default="", type=str, help='The checkpoint model path')
    parser.add_argument('--save_config', default=False, action='store_true', help='Save paramters config file.')
    parser.add_argument('--load_config', default=False, action='store_true', help='Load parametes config file.')
    parser.add_argument('--config_path', default="", type=str, help='The config file path.')
    parser.add_argument('--freq_init', default=4, type=int)
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--shuffle', default=True, action='store_false')
    parser.add_argument('--norm_flag', default=False, action='store_true')
    parser.add_argument('--bern_flag', default=False, action='store_true')

    #parser only for HAKE
    parser.add_argument("--phase_weight", default=1.0, type=float, help='only on HAKE, The weight of phase part')
    parser.add_argument("--modulus_weight", default=1.0, type=float, help='only on HAKE, The weight of modulus part')
    
    #parser only for knn
    parser.add_argument("--calculate_knn", default=False, action='store_true', help='calculate k nearest neighbor in three level')
    parser.add_argument("--base_model_name", type=str)
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--ent_knn", default=1, type=int)
    parser.add_argument("--rel_knn", default=1, type=int)
    parser.add_argument("--triple_knn", default=1, type=int)
    #parser.add_argument("--lambda_knn", default=[], type=float, nargs='+')
    parser.add_argument("--ent_lambda", default=0.1, type=float)
    parser.add_argument("--rel_lambda", default=0.1, type=float)
    parser.add_argument("--triple_lambda", default=0.1, type=float)
    parser.add_argument("--triple_ent_knn", default=500, type=int)
    parser.add_argument("--triple_rel_knn", default=5, type=int)
    parser.add_argument("--trans_alpha", default=1.0, type=float)
    parser.add_argument("--alpha", default=10.0, type=float)
    parser.add_argument("--set_level", default=0, type=int)
    parser.add_argument("--set_loss_distance", default=0, type=int)
    parser.add_argument("--milestones", default=[50, 200], type=int, nargs='+')
    parser.add_argument("--anafunc", default='norm', type=str)
    parser.add_argument("--use_pickle", default=False, action='store_true')
    parser.add_argument("--case_study", default=False, action='store_true')

    #parser for loggging
    parser.add_argument("--save_path", type=str, default='logging')
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--special_name",type=str, default=None)

    # Get data, model, and LitModel specific arguments
    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model.BaseLitModel.add_to_argparse(lit_model_group)

    data_group = parser.add_argument_group("Data Args")
    data.BaseDataModule.add_to_argparse(data_group)


    parser.add_argument("--help", "-h", action="help")
    
    
    return parser
