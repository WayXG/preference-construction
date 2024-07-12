# preference-construction

## Get Data

```
In one terminal:

export RAY_memory_monitor_refresh_ms=0; python -m vllm.entrypoints.api_server     --model /home/guhao/xw/rsf/rsf_hf_mistral_rsf_baseline/model1     --gpu-memory-utilization=0.9     --max-num-seqs=200     --host 0.0.0.0 --tensor-parallel-size 4     --port 8000


After running the first command, in another terminal:

python gen_data_vllm_with_filter.py --tokenizer HuggingFaceH4/mistral-7b-sft-beta --dataset_name_or_path /home/guhao/xw/ultra/data/iter1.json --output_dir /home/guhao/xw/ultra/data/iter2.json --K 1 --max_new_tokens 2048
```
