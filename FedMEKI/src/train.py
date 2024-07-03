import argparse
import deepspeed
import json
import logging
import numpy as np
import os
import random
import time
import torch
from tqdm import tqdm
from transformers.deepspeed import HfDeepSpeedConfig
import yaml

from model import load_model
from datasets import load_dataset
from copy import deepcopy
from local_FL import * 

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def parser_args():
    parser = argparse.ArgumentParser(description="train parameters for LAMM")
    parser.add_argument(
        "--cfg", type=str, default="./config/train.yaml", help="config file"
    )
    # data-related configurations
    parser.add_argument(
        "--data_path",
        type=str,
        # required=True,
        help="the path that stores the data JSON",
    )
    parser.add_argument(
        "--vision_root_path", 
        type=str, 
        # required=True, 
        help="Root dir for images"
    )
    parser.add_argument(
        "--max_tgt_len",
        type=int,
        default=400,
        help="max length of post-image texts in LLM input",
    )
    parser.add_argument(
        "--vision_type",
        type=str,
        default="image",
        choices=("image", "pcl"),
        help="the type of vision data",
    )
    parser.add_argument(
        "--use_system",
        default=False,
        action="store_true",
        help="whether to use system messages",
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--save_path", type=str, help="directory to save checkpoints")
    parser.add_argument("--log_path", type=str, help="directory to save logs")
    # model-related configurations
    parser.add_argument(
        "--model", type=str, default="lamm_peft", help="Model class to use"
    )
    parser.add_argument(
        "--encoder_pretrain",
        type=str,
        default="clip",
        choices=("clip", "epcl"),
        help="Vision Pretrain Model",
    )
    parser.add_argument(
        "--encoder_ckpt_path",
        type=str,
        help="path of vision pretrained model; CLIP use default path in cache",
    )
    parser.add_argument(
        "--llm_ckpt_path",
        type=str,
        required=True,
        help="path of LLM, default: Vicuna",
    )
    parser.add_argument(
        "--delta_ckpt_path",
        type=str,
        help="path of delta parameters from previous stage; Only matter for stage 2",
    )
    parser.add_argument(
        "--llm_proj_path",
        type=str,
        help="path of LLM projection matrix; Only matter for stage 2",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    # embedding configurations
    parser.add_argument(
        "--vision_feature_type",
        type=str,
        default="local",
        choices=("local", "global"),
        help="the type of vision features",
    )
    parser.add_argument(
        "--vision_output_layer",
        type=int,
        default=-1,
        choices=(-1, -2),
        help="the layer to output visual features; -1 means global from last layer",
    )
    parser.add_argument("--num_vision_token", type=int, default=1, help="number of vision tokens")
    parser.add_argument("--conv_template", type=str, default="default", help="which conversation template to use")
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="number of training stage; 1 by default; 2 if delta ckpt specified",
    )
    # PCL configurations
    parser.add_argument(
        "--use_color",
        default=False,
        action="store_true",
        help="whether to use color of point cloud",
    )
    parser.add_argument(
        "--use_height",
        default=False,
        action="store_true",
        help="whether to use height info of point cloud",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=40000,
        help="number of points in each point cloud",
    )
    # flash attention
    parser.add_argument(
        "--use_flash_attn",
        default=False,
        action="store_true",
        help="whether to use flash attention to speed up",
    )
    # xformers
    parser.add_argument(
        "--use_xformers",
        default=False,
        action="store_true",
        help="whether to use xformers to speed up",
    )
    parser.add_argument(
        "--num_clients",
        default=5,
        # action="store_true",
        help="number of clients in federated knowledge injection",
    )
    args = parser.parse_args()

    assert not (args.use_flash_attn and args.use_xformers), 'can only use one of flash attn and xformers.'

    # if args.vision_feature_type == "local":
    #     args.num_vision_token = 198
    #     args.vision_output_layer = -2
    # elif args.vision_feature_type == "global":
    #     args.num_vision_token = 1
    #     args.vision_output_layer = -1
    # else:
    #     raise NotImplementedError(
    #         "NOT implement vision feature type: {}".format(args.vision_feature_type)
    #     )

    print(
        "Arguments: \n{}".format(
            json.dumps(vars(parser.parse_args()), indent=4, sort_keys=True)
        )
    )
    return args


def initialize_distributed(args):
    args["master_ip"] = os.getenv("MASTER_ADDR", "localhost")
    args["master_port"] = os.getenv("MASTER_PORT", "6000")
    args["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    args["local_rank"] = 0 % torch.cuda.device_count()
    os.environ['LOCAL_RANK']='0'
    device = args["local_rank"] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args["root_dir"] = "../"
    args["mode"] = "train"
    initialize_distributed(args)
    set_random_seed(args["seed"])


def build_directory(path):
    if os.path.exists(path):
        pass
    else:  # recursively construct directory
        os.makedirs(path, exist_ok=True)

def clip_to_float16_range(tensor):
    FLOAT16_MAX = 65504.0
    FLOAT16_MIN = 6.1e-5
    if tensor.dim() == 0:  # Handle scalar tensor
        if tensor != tensor:  # Check for NaN
            return torch.tensor(0.0, dtype=tensor.dtype)
        elif torch.isinf(tensor):  # Check for Inf
            return torch.tensor(FLOAT16_MAX if tensor > 0 else -FLOAT16_MAX, dtype=tensor.dtype)
        else:
            return torch.clamp(tensor, -FLOAT16_MAX, FLOAT16_MAX)
    else:
        tensor = torch.clamp(tensor, -FLOAT16_MAX, FLOAT16_MAX)
        tensor[(tensor != tensor).nonzero(as_tuple=True)] = 0.0  # Remove NaNs
        tensor[torch.isinf(tensor)] = FLOAT16_MAX  # Remove Infs
        return tensor

def main(**args):
    start_time = time.time()
    config_env(args)
    build_directory(args["save_path"])
    build_directory(args["save_path"] + '/' + str(args["num_clients"]))
    build_directory(args["log_path"])

    # dump training settings
    with open(os.path.join(args["log_path"], "training_args.json"), "w") as fw:
        json.dump(args, fw, indent=4)

    dschf = HfDeepSpeedConfig(args["deepspeed"])
    args["dschf"] = dschf

    if args["log_path"]:
        logging.basicConfig(
            format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode="w",
        )
    
    if args['use_flash_attn']:
        from model.LAMM.flash_attn_patch import replace_llama_attn_with_flash_attn
        logging.info("⚡⚡⚡ enable flash attention.")
        replace_llama_attn_with_flash_attn()

    if args['use_xformers']:
        from model.LAMM.xformers_patch import replace_llama_attn_with_xformers_attn
        logging.info("xxx enable xformers attention.")
        replace_llama_attn_with_xformers_attn()



    server_rsna_data_path = path_to_server_rsna_data  
    server_covid_data_path = path_to_server_covid_data  
    server_ecg_data_path = path_to_server_ecg_data  
    server_clinical_data_path = path_to_server_clinical_data 
    
    client_rsna_data_path = path_to_client_rsna_data
    client_covid_data_path = path_to_client_covid_data
    client_ecg_data_path = path_to_client_ecg_data
    client_clinical_data_path = path_to_client_clinical_data
    
    test_rsna_data_path = path_to_test_rsna_data
    test_covid_data_path = path_to_test_covid_data
    test_ecg_data_path = path_to_test_ecg_data
    test_clinical_data_path = path_to_test_clinical_data

    
    
    # Load the datasets
    RSNA_train_data, RSNA_train_iter, _ = load_dataset(args, server_rsna_data_path)
    covid_train_data, covid_train_iter, _ = load_dataset(args, server_covid_data_path)
    mortality_train_data, mortality_train_iter, _ = load_dataset(args, server_ecg_data_path)
    ecg_train_data, ecg_train_iter, _ = load_dataset(args, server_clinical_data_path)

    # Combine all data loaders
    train_iters = [
        
        ('mortality', mortality_train_iter),
        ('ecg', ecg_train_iter),
        ('image', RSNA_train_iter),
        ('covid', covid_train_iter)
    ]
  
    modalities = ['image', 'covid', 'ecg', 'clinicals']
    loaders = {}
    for modality in modalities:
        if modality == 'image':
            image_paths, image_labels = parse_data(client_rsna_data_path, 'image')
            test_image_paths, test_image_labels = parse_data(test_rsna_data_path, 'image')
            loaders['train_image'] = DataLoader(ImageDataset(image_paths, image_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=True)
            loaders['test_image'] = DataLoader(ImageDataset(test_image_paths, test_image_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=False)
        elif modality == 'covid':
            covid_paths, covid_labels = parse_data(client_covid_data_path, 'covid')
            test_covid_paths, test_covid_labels = parse_data(test_covid_data_path, 'covid')
            loaders['train_covid'] = DataLoader(ImageDataset(covid_paths, covid_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=True)
            loaders['test_covid'] = DataLoader(ImageDataset(test_covid_paths, test_covid_labels, AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')), batch_size=32, shuffle=False)
        elif modality == 'ecg':
            ecg_paths, ecg_labels = parse_data(client_ecg_data_path, 'ecg')
            test_ecg_paths, test_ecg_labels = parse_data(test_ecg_data_path, 'ecg')
            loaders['train_ecg'] = DataLoader(ECGDataset(ecg_paths, ecg_labels), batch_size=32, shuffle=True)
            loaders['test_ecg'] = DataLoader(ECGDataset(test_ecg_paths, test_ecg_labels), batch_size=32, shuffle=False)
        elif modality == 'clinicals':
            clinical_data, clinical_labels = parse_data(client_clinical_data_path, 'clinicals')
            test_clinical_data, test_clinical_labels = parse_data(test_clinical_data_path, 'clinicals')
            loaders['train_clinicals'] = DataLoader(ClinicalDataset(clinical_data, clinical_labels), batch_size=32, shuffle=True)
            loaders['test_clinicals'] = DataLoader(ClinicalDataset(test_clinical_data, test_clinical_labels), batch_size=32, shuffle=False)

    
    
    
    # Calculate the length and total steps
    length = (
        args["epochs"]
        * sum(len(data) for data in [RSNA_train_data, covid_train_data, mortality_train_data, ecg_train_data])
        # * sum(len(data) for data in [RSNA_train_data])
        // args["world_size"]
        // dschf.config["train_micro_batch_size_per_gpu"]
    )
    total_steps = args["epochs"] * sum(len(data) for data in [RSNA_train_data, covid_train_data, mortality_train_data, ecg_train_data]) // dschf.config["train_batch_size"]
    args["total_steps"] = total_steps

    # Load the model
    agent = load_model(args)
    




    # Barrier for distributed training
    torch.distributed.barrier()

    # Save training arguments
    with open(os.path.join(args["log_path"], "training_args.yaml"), "w") as fw:
        yaml.dump(args, fw)

    # Initialize progress bar
    pbar = tqdm(total=length)  # maximum total number
    current_step = 0


    # Specify the modalities you would like to cover    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    
    for param in agent.model.visual_encoder.parameters():
        param.requires_grad = True
    for param in agent.model.signal_module.parameters():
        param.requires_grad = True
    for param in agent.model.clinical_module.parameters():
        param.requires_grad = True
    for param in agent.model.llama_proj.parameters():
        param.requires_grad = True
        
        
    # Training loop
    for epoch_i in tqdm(range(args["epochs"])):
        # Federated learning: local training
        if int(args["num_clients"]) != 0:
            if "local_model" in locals():
                new_state_dict_fp32 = deepcopy(agent.model.visual_encoder.state_dict())
                new_state_dict_fp16 = {key: value.float() for key, value in new_state_dict_fp32.items()}
                local_model.visual_encoder.load_state_dict(new_state_dict_fp16)
                new_state_dict_fp32 = deepcopy(agent.model.signal_module.state_dict())
                new_state_dict_fp16 = {key: value.float() for key, value in new_state_dict_fp32.items()}
                local_model.signal_module.load_state_dict(new_state_dict_fp16)
                new_state_dict_fp32 = deepcopy(agent.model.clinical_module.state_dict())
                new_state_dict_fp16 = {key: value.float() for key, value in new_state_dict_fp32.items()}
                local_model.clinical_module.load_state_dict(new_state_dict_fp16)
                local_model = federated_training(
                    image_loader=loaders.get('train_image'),
                    covid_loader=loaders.get('train_covid'),
                    ecg_loader=loaders.get('train_ecg'),
                    clinical_loader=loaders.get('train_clinicals'),
                    use_server_data=False,
                    server_model=local_model,
                    epochs=1,
                    use_amp=True,
                    num_clients=5,
                    federate_learning=True,
                    device=device
                )
            else:
                local_model = federated_training(
                    image_loader=loaders.get('train_image'),
                    covid_loader=loaders.get('train_covid'),
                    ecg_loader=loaders.get('train_ecg'),
                    clinical_loader=loaders.get('train_clinicals'),
                    use_server_data=False,
                    epochs=1,
                    use_amp=True,
                    num_clients=5,
                    federate_learning=True,
                    device=device
                )

            local_model.to(agent.model.device)
            local_state_dict_fp32 = deepcopy(local_model.visual_encoder.to(agent.model.device).state_dict())
            local_state_dict_fp16= {key: clip_to_float16_range(value) for key, value in local_state_dict_fp32.items()}            
            agent.model.visual_encoder.load_state_dict(local_state_dict_fp16)
            agent.model.visual_encoder.to(agent.model.device)
    
            # Signal Module
            local_state_dict_fp32 = deepcopy(local_model.signal_module.to(agent.model.device).state_dict())
            local_state_dict_fp16 = {key: clip_to_float16_range(value) for key, value in local_state_dict_fp32.items()}            
            agent.model.signal_module.load_state_dict(local_state_dict_fp16)
            agent.model.signal_module.to(agent.model.device)
    
            # Clinical Module
            local_state_dict_fp32 = deepcopy(local_model.clinical_module.to(agent.model.device).state_dict())
            local_state_dict_fp16 = {key: clip_to_float16_range(value) for key, value in local_state_dict_fp32.items()}                    
            agent.model.clinical_module.load_state_dict(local_state_dict_fp16)
            agent.model.clinical_module.to(agent.model.device)

        for modality, train_iter in train_iters:



            agent.ds_engine.zero_grad()
            agent.optimizer.zero_grad()
            # Iterate over batches in the current data loader
            for batch in train_iter:
                agent.train_model(batch, current_step=current_step, pbar=pbar)
                current_step += 1
            torch.distributed.barrier()

        # Save the model at specific intervals
        if epoch_i % max(args["epochs"] // 5, 1) == 0:  # save epoch1 & save 5 models at most
            agent.save_model(args["save_path"], epoch_i + 1)
            if int(args["num_clients"]) != 0:
              save_path = args["save_path"] + '/' + str(args["num_clients"])
              torch.save(local_model, save_path + '/' + "local.pt")



    save_path = args["save_path"] + '/' + str(args["num_clients"])
    if int(args["num_clients"]) != 0:
        torch.save(local_model, save_path + '/' + "local.pt")
    torch.distributed.barrier()
    agent.save_model(save_path, 0)

    print(f"Done! Total Training time: {time.time() - start_time}")

if __name__ == "__main__":
    args = parser_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    args = vars(args)
    # arguments from command line have higher priority
    cfg.update(args)
    main(**cfg)
