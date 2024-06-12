#!/bin/bash

# Array of specific combinations of delta_ckpt_path and base_data_path
combinations=(
    # "/data/xiaochen/FedMFM/ckpt/ours_fedprox_multi__/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_few_shot_test.json"

    #. "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/chexpert_few_shot_test.json"
    # "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/sepsis_few_test.json"
    # "/data/xiaochen/FedMFM/ckpt/ours_fedprox_multi__/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/chexpert_few_shot_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/rad_few_shot_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/ecgqa_few_shot_test.json"
    
    # "/data/xiaochen/FedMFM/ckpt/ours_fedprox_multi__/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/sepsis_few_test.json"
#    "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/RSNA_test.json"
#    "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/covid_test.json"
#    "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/ecg_test.json"
#    "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/mortality_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_no_fl/0/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/covid_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_no_modal/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/covid_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_no_task/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/covid_test.json"
    # Add more combinations as needed
    # "/data/xiaochen/FedMFM/ckpt/fedprox_multi_/pytorch_model_ep3.pt /data/xiaochen/FedMFM/preprocessed_jsons/eb_test.json"
    "/data/xiaochen/FedMFM/ckpt/fedavg_multi_/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/slake_test.json"
    #"/data/xiaochen/FedMFM/ckpt/fedprox_multi_/pytorch_model_ep3.pt /data/xiaochen/FedMFM/preprocessed_jsons/ate_test.json"
    #"/data/xiaochen/FedMFM/ckpt/fedprox_multi_/pytorch_model_ep3.pt /data/xiaochen/FedMFM/preprocessed_jsons/pf_test.json"
    #"/data/xiaochen/FedMFM/ckpt/fedprox_multi_/pytorch_model_ep3.pt /data/xiaochen/FedMFM/preprocessed_jsons/slake_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_fedprox_multi__/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/slake_test.json"
    # "/data/xiaochen/FedMFM/ckpt/ours_fedavg_multi___/5/pytorch_model.pt /data/xiaochen/FedMFM/preprocessed_jsons/slake_test.json"
    #"/data/xiaochen/FedMFM/ckpt/ours_fedprox_multi__/5/pytorch_model /data/xiaochen/FedMFM/preprocessed_jsons/slake_test.json"
    
    
    
)
# Loop over each combination of delta_ckpt_path and base_data_path
for combo in "${combinations[@]}"; do
    # Split the combination into delta_ckpt_path and base_data_path
    delta_ckpt_path=$(echo $combo | cut -d' ' -f1)
    base_data_path=$(echo $combo | cut -d' ' -f2)

    # Update the model configuration with the current delta_ckpt_path
    cat << EOF > temp_model_config.yaml
model_name: Octavius_2d
stage: 2
octavius_modality: ['signal']

llm_ckpt_path: /data/xiaochen/FedMFM/MMedLM2/
delta_ckpt_path: "${delta_ckpt_path}"

encoder_pretrain: clip
vision_feature_type: local
num_vision_token: 198

peft_type: moe_lora
moe_lora_num_experts: 1
moe_gate_mode: top2_gate
lora_r: 8
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: ['wqkv', 'wo', 'w1', 'w2', 'w3', 'output']

num_query_rsp_3d: 16
hidden_size_rsp_3d: 768
num_layers_rsp_3d: 1
num_heads_rsp_3d: 8

max_tgt_len: 100
conv_mode: simple
EOF

    # Update the dataset configuration with the current base_data_path
    cat << EOF > temp_dataset_config.yaml
scenario_cfg:
  dataset_name: RSNA
  base_data_path: "${base_data_path}"

eval_cfg:
  instruction_cfg:
    query_type: standard_query
  inferencer_cfg:
    inferencer_type: Direct
    batch_size: 128
  metric_cfg:
    metric_type: LAMM
EOF

    # Run the evaluation
    python eval.py \
        --model_cfg temp_model_config.yaml \
        --recipe_cfg temp_dataset_config.yaml

done

# Clean up temporary files
rm temp_model_config.yaml
rm temp_dataset_config.yaml
