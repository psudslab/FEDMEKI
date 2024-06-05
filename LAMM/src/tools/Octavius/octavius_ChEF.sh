# choose one model_cfg from 2d, 3d, 2d+3d
model_cfg=config/ChEF/models/octavius_2d.yaml
dataset=FedKI

python eval.py \
            --model_cfg ${model_cfg} \
            --recipe_cfg config/ChEF/scenario_recipes/LAMM/${dataset}.yaml
