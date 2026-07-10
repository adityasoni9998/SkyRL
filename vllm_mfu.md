# Starting the vLLM server

1. Enter apptainer first.
```bash
apptainer shell \
    --hostname "$(hostname -s)" \
    --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem,/etc/pki/tls/certs:/etc/pki/tls/certs" \
    /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif 
```

2. Activate env inside apptainer
```bash
source /work1/grahamneubig/adityabs/.skyrl_venv/bin/activate
```

3. Launch vLLM server with different config options
- NOTE: Always compute avg. MFU metrics and throughputs for different config options.
```bash
HOME=/tmp vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 2048 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --port 8001 \
  --enable-mfu-metrics
```

If you want to try the ROCM_AITER_FA backend, use this command:

```bash
VLLM_ROCM_USE_AITER=1 HOME=/tmp vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 1024 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --port 8001 \
  --enable-mfu-metrics \
  --attention-backend ROCM_AITER_FA
```

# Launching the vLLM workload script

1. Enter apptainer first.
```bash
apptainer shell \
    --hostname "$(hostname -s)" \
    --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem,/etc/pki/tls/certs:/etc/pki/tls/certs" \
    /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif 
```

2. Activate env inside apptainer
```bash
source /work1/grahamneubig/adityabs/.skyrl_venv/bin/activate
```

3. Run the workload script with different prefix hit rates and qps
```bash
python3 scripts/run_preset_vllm_workload.py \
  --base-url http://127.0.0.1:8001 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --duration-s 180 \
  --qps 1.5 \
  --max-concurrency 1200000000000 \
  --prefix-hit-rate 0.30 
```

# Running the sweep

```bash
cd /work1/grahamneubig/adityabs/skyrl_max_tokens_per_microbatch
DURATION_S=120 bash scripts/run_vllm_mfu_sweep.sh
```
1. ROCM_AITER_FA is important - lower latency.
2. enforce_eager matters a lot
3. max_num_seqs, and max_num_batched_tokens don't matter: 512, 131072 are good enough
  a. Prefill-heavy workload - try larger max_num_batched_tokens if it helps.
4. qps improves throughput and latency both.