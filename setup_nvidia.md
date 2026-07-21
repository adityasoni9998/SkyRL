source ../.skyrl_venv/bin/activate
rm -rf skyrl/tinker/tinker.db*
rm -rf /tmp/skyrl*
mkdir -p /tmp/skyrl_tinker
mkdir -p /tmp/skyrl_checkpoints
export HOME=/tmp
export SKYRL_RAY_NUM_CPUS=192
HOME=/tmp SKYRL_RAY_NUM_CPUS=192 uv run --active --no-sync --extra tinker --extra fsdp \
    -m skyrl.tinker.api \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --backend fsdp \
    --port 9000 \
    --checkpoints-base /tmp/skyrl_checkpoints \
    --database-url sqlite:////tmp/skyrl_tinker/tinker.db \
    --backend-config '{
        "trainer.placement.colocate_all": false,
        "trainer.placement.policy_num_nodes": 1,
        "trainer.placement.policy_num_gpus_per_node": 4,
        "trainer.max_tokens_per_microbatch": 65000,
        "trainer.use_expandable_segments": true,

        "generator.inference_engine.num_engines": 4,
        "generator.inference_engine.max_num_batched_tokens": 131072,
        "generator.inference_engine.enable_ray_prometheus_stats": true,
        "generator.inference_engine.engine_init_kwargs.enable_mfu_metrics": true,
        "generator.inference_engine.gpu_memory_utilization": 0.8,
        "generator.inference_engine.max_num_seqs": 256,
        "generator.inference_engine.enable_prefix_caching": false,
        "generator.inference_engine.enforce_eager": true,
        "generator.inference_engine.engine_init_kwargs.max_model_len": 32768,
        "generator.inference_engine.use_expandable_segments": false
    }'