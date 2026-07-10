# Tinker Server Startup Instructions for AMD GPUs

## Launch Apptainer Shell

```bash
cd /work1/grahamneubig/adityabs/skyrl_max_tokens_per_microbatch
apptainer shell \
    --hostname "$(hostname -s)" \
    --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem,/etc/pki/tls/certs:/etc/pki/tls/certs" \
    /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif 
```

## Launch the Tinker Server
```bash
source /work1/grahamneubig/adityabs/.skyrl_venv/bin/activate
mkdir -p /tmp/skyrl-tinker
ray stop --force

export VLLM_ROCM_USE_AITER=1
export HOME=/tmp
export SKYRL_RAY_NUM_CPUS=256
uv run --active --no-sync --extra tinker --extra fsdp \
    -m skyrl.tinker.api \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --backend fsdp \
    --port 9000 \
    --database-url sqlite:////tmp/skyrl-tinker/tinker.db \
    --checkpoints-base /work1/grahamneubig/adityabs/skyrl_checkpoints \
    --backend-config '{
        "trainer.placement.colocate_all": false,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 4,
        "trainer.max_tokens_per_microbatch": 96000,
        "trainer.use_expandable_segments": false,

        "generator.inference_engine.num_engines": 4,
        "generator.inference_engine.max_num_batched_tokens": 65536,
        "generator.inference_engine.enable_ray_prometheus_stats": true,
        "generator.inference_engine.engine_init_kwargs.enable_mfu_metrics": true,
        "generator.inference_engine.gpu_memory_utilization": 0.9,
        "generator.inference_engine.max_num_seqs": 256,
        "generator.inference_engine.enforce_eager": false,
        "generator.inference_engine.engine_init_kwargs.max_model_len": 32768,
        "generator.inference_engine.engine_init_kwargs.attention_backend": "ROCM_AITER_FA",
        "generator.inference_engine.use_expandable_segments": false
    }'
```

## Forward Port to Babel
- NOTE: Edit /home1/adityabs/.ssh/config and add the right babel-compute-node name if needed.
```bash
ssh -N -R 0.0.0.0:9000:localhost:9000 babel-compute-node
```
