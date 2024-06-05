import datetime
import os
import yaml

from ChEF.models import get_model
from ChEF.scenario import dataset_dict
from ChEF.evaluator import Evaluator, load_config, sample_dataset

import random
import torch
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def main():
    model_cfg, recipe_cfg, save_dir, sample_len = load_config()
    # model
    model = get_model(model_cfg)
    # dataset
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    print(dataset_name)
    print(dataset_dict)
    dataset = dataset_dict['FedKI'](**scenario_cfg)
    # sample dataset
    # dataset = sample_dataset(dataset, sample_len=100, sample_seed=0)
    # print(dataset)
    # save_cfg
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'], dataset_name, time)
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg), stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')

    # evaluate
    eval_cfg = recipe_cfg['eval_cfg']
    evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
    evaluater.evaluate(model)

if __name__ == '__main__':
    main()