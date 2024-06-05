#!/bin/bash

num_clients_values=(5)
modalities=("image" "medqa" "ecg" "mortality" "ecgqa") # Add other modalities as needed
now=$(date +"%Y%m%d_%H%M%S")

for num_clients in "${num_clients_values[@]}"
do
    for modality in "${modalities[@]}"
    do
        python -m torch.distributed.launch --nproc_per_node=1 --master_port=2334 train_single.py \
            --stage 1 \
            --cfg config/Octavius/octavius_2d_e4_bs64.yaml \
            --conv_template default \
            --max_tgt_len 128 \
            --use_system \
            --model octavius \
            --encoder_pretrain clip \
            --llm_ckpt_path /data/xiaochen/FedMFM/MMedLM2/ \
            --vision_feature_type local \
            --num_vision_token 198 \
            --save_path /data/xiaochen/FedMFM/ckpt/fedavg_${modality} \
            --log_path ../ckpt/octavius_2d_e4_bs64/log_rest/ \
            --num_clients $num_clients \
            --modalities $modality \
            2>&1 | tee ../ckpt/octavius_2d_e4_bs64/log_rest/train_${now}_clients${num_clients}_${modality}.log
    done
done
