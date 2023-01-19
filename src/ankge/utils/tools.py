import importlib
from IPython import embed
import logging
import json
import datetime
import os
import time
import yaml

def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'model.TransE'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def save_config(args):
    args.save_config = False  
    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime(str(args.model_name)+"_"+str(args.dataset_name)) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.makedirs(os.path.join("config", day_name))
    config = vars(args)
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

def load_config(args, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        args.__dict__.update(config)
    return args


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    date_file = os.path.join(args.save_path, date)

    if not os.path.exists(date_file):
        os.makedirs(date_file)

    hour = str(int(dt.strftime("%H")) + 8)
    name = hour + dt.strftime("_%M_%S")
    if args.special_name != None:
        name = args.special_name
    log_file = os.path.join(date_file,  "_".join([args.model_name, args.dataset_name, name, 'train.log']))

    logging.basicConfig(
        format='%(asctime)s %(message)s',
        level=logging.INFO,
        datefmt='%m-%d %H:%M',
        filename=log_file,
        filemode='a'
    )

def log_metrics(epoch, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s: %.4f at epoch %d' % (metric, metrics[metric], epoch))


def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']