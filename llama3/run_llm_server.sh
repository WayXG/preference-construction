export LD_LIBRARY_PATH=/home/nilgeoutim/anaconda3/envs/llm/lib/python3.9/site-packages/nvidia/nvjitlink/lib
export HUGGING_FACE_HUB_TOKEN=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
export CUDA_VISIBLE_DEVICES=2
export RAY_memory_monitor_refresh_ms=0

seed=37
tensor_parallel_size=1

python -m vllm.entrypoints.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 200 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size ${tensor_parallel_size} \
    --seed ${seed} \