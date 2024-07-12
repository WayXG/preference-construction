export LD_LIBRARY_PATH=/home/nilgeoutim/anaconda3/envs/llm/lib/python3.9/site-packages/nvidia/nvjitlink/lib
export HUGGING_FACE_HUB_TOKEN=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
export CUDA_VISIBLE_DEVICES=2

role_num=38
task_num=1000
# seeds=(37 42 0)
seeds=(37)
temperature=0.1

# for seed in ${seeds[@]}; do
#     python get_responses_llama3.py \
#         --output_dir_pref /data-3/common/PrefCon/temperature0/role${role_num}task${task_num}seed${seed}.json \
#         --output_dir_data /data-3/common/PrefCon/temperature0/task${task_num}seed${seed}.json \
#         --role_num ${role_num} \
#         --task_num ${task_num} \
#         --seed ${seed} \
#         --temperature ${temperature}
# done

# for seed in ${seeds[@]}; do
#     python get_responses_llama3.py \
#         --dataset_name_or_path /data-3/common/PrefCon/temperature0/task${task_num}seed${seed}.json \
#         --reverse 1 \
#         --output_dir_pref /data-3/common/PrefCon/temperature0/role${role_num}task${task_num}seed${seed}reversed.json \
#         --output_dir_data /data-3/common/PrefCon/temperature0/task${task_num}seed${seed}reversed.json \
#         --role_num ${role_num} \
#         --task_num ${task_num} \
#         --seed ${seed} \
#         --temperature ${temperature}
# done

for seed in ${seeds[@]}; do
    python get_scores_llama3.py \
        --output_dir_pref /data-3/common/PrefCon/score_temp0.1/role${role_num}task${task_num}seed${seed}.json \
        --output_dir_data /data-3/common/PrefCon/score_temp0.1/task${task_num}seed${seed}.json \
        --role_num ${role_num} \
        --task_num ${task_num} \
        --seed ${seed} \
        --temperature ${temperature}
done

for seed in ${seeds[@]}; do
    python get_scores_llama3.py \
        --dataset_name_or_path /data-3/common/PrefCon/score_temp0.1/task${task_num}seed${seed}.json \
        --reverse 1 \
        --output_dir_pref /data-3/common/PrefCon/score_temp0.1/role${role_num}task${task_num}seed${seed}reversed.json \
        --output_dir_data /data-3/common/PrefCon/score_temp0.1/task${task_num}seed${seed}reversed.json \
        --role_num ${role_num} \
        --task_num ${task_num} \
        --seed ${seed} \
        --temperature ${temperature}
done