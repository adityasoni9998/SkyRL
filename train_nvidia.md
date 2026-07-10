# Tinker Server Startup Instructions for NVIDIA GPUs

```bash
source ../.venv/bin/activate
ray stop --force
rm -rf /tmp/skyrl-tinker
mkdir -p /tmp/skyrl-tinker
mkdir -p /tmp/skyrl_checkpoints
export SKYRL_DISABLE_VLLM_CLEAR_CUDA_CACHE=1
SKYRL_DISABLE_VLLM_CLEAR_CUDA_CACHE=1 HOME=/tmp uv run --active --no-sync --extra tinker --extra fsdp \
    -m skyrl.tinker.api \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --backend fsdp \
    --port 9000 \
    --database-url sqlite:////tmp/skyrl-tinker/tinker.db \
    --checkpoints-base /tmp/skyrl_checkpoints \
    --max-checkpoints-to-keep 3 \
    --backend-config '{
        "trainer.placement.colocate_all": false,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 4,
        "trainer.max_tokens_per_microbatch": 65536,

        "generator.inference_engine.num_engines": 4,
        "generator.inference_engine.max_num_batched_tokens": 131072,
        "generator.inference_engine.max_num_seqs": 512,
        "generator.inference_engine.enable_ray_prometheus_stats": true,
        "generator.inference_engine.engine_init_kwargs.enable_mfu_metrics": true,
        "generator.inference_engine.gpu_memory_utilization": 0.8,
        "generator.inference_engine.enforce_eager": false,
        "generator.inference_engine.engine_init_kwargs.max_model_len": 32768
    }'
```

```bash
ssh -N -R 0.0.0.0:9000:localhost:9000 babel-compute-node
```
