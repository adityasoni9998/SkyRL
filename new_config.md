data:                  
  dataloader:          
    num_workers: 8     
    persistent_workers: false                                    
  train_data:          
  - /tmp/data/gsm8k/train.parquet                                
  val_data:            
  - /tmp/data/gsm8k/validation.parquet                           
environment:           
  env_class: gsm8k     
  skyrl_gym:           
    llm_as_a_judge:    
      base_url: null   
      model: gpt-4o-mini                                         
    max_env_workers: 32
    search:            
      log_requests: false                                        
      search_url: http://127.0.0.1:8000/retrieve                 
      timeout: 30      
      topk: 3          
    text2sql:          
      db_path: /home/ray/default/sql_data                        
generator:             
  append_eos_token_after_stop_str_in_multi_turn: true            
  apply_overlong_filtering: false                                
  batched: false       
  chat_template:       
    name_or_path: null 
    source: name       
  chat_template_kwargs: {}                                       
  eval_n_samples_per_prompt: 1                                   
  eval_sampling_params:
    additional_kwargs: null                                      
    logprobs: 1        
    max_generate_length: 1024                                    
    min_p: 0.0         
    repetition_penalty: 1.0                                      
    stop: null         
    temperature: 0.0   
    top_k: -1          
    top_p: 1.0         
  inference_engine:                                     
    async_engine: true                                  
    backend: vllm                                       
    data_parallel_size: 1                               
    distributed_executor_backend: ray                   
    enable_chunked_prefill: true                        
    enable_http_endpoint: false                         
    enable_pd: false                                    
    enable_prefix_caching: true                         
    enable_ray_prometheus_stats: false                  
    enable_return_routed_experts: false                 
    enforce_eager: true                                 
    engine_init_kwargs:                                 
      max_model_len: 32768                              
    expert_parallel_size: 1                             
    external_proxy_url: null                            
    external_server_urls: null                          
    fully_sharded_loras: false                          
    gpu_memory_utilization: 0.8                         
    http_endpoint_host: 127.0.0.1                       
    http_endpoint_port: 8000                            
    language_model_only: false                          
    max_num_batched_tokens: 131072                      
    max_num_seqs: 1024                                  
    model_dtype: bfloat16                               
    num_engines: 6                                      
    num_prefill: 0                                      
    override_existing_update_group: auto                
    pipeline_parallel_size: 1                           
    remote_urls: []
    router_init_kwargs: {}                
    run_engines_locally: true             
    served_model_name: null               
    tensor_parallel_size: 1               
    use_expandable_segments: false        
    vllm_v1_disable_multiproc: true       
    weight_sync_backend: nccl             
    weight_transfer_threshold_cuda_ipc_GB: 1.0                                      
  max_input_length: 512                   
  max_turns: 1                            
  merge_stepwise_output: false            
  n_samples_per_prompt: 5                 
  rope_scaling: null                      
  rope_theta: null                        
  sampling_params:                        
    additional_kwargs: null               
    logprobs: 1                           
    max_generate_length: 1024             
    min_p: 0.0                            
    repetition_penalty: 1.0               
    stop: null                            
    temperature: 1.0                      
    top_k: -1                             
    top_p: 1.0                            
  step_wise_trajectories: false           
  use_conversation_multi_turn: true       
  vision_language_generator: false        
  zero_reward_on_non_stop: false          
trainer:                                  
  algorithm:                              
    advantage_batch_normalize: false      
    advantage_estimator: grpo             
    cispo:                                
      cispo_eps_clip_high: 5.0            
      cispo_eps_clip_low: 0.0             
    clip_cov:                             
      clip_cov_lb: 1.0                    
      clip_cov_ub: 5.0                    
      clip_ratio: 0.0002                  
    clip_ratio_c: 3.0                     
    dynamic_sampling:                     
      max_sample_batches: 30              
      min_replace_ratio: 0.3              
      type: null                          
    entropy_loss_coef: 0.01                             
    eps_clip_high: 0.2                                  
    eps_clip_low: 0.2                                   
    gamma: 1.0
    grpo_norm_by_std: true                              
    kl_cov:   
      kl_cov_frac: 0.2                                  
      ppo_kl_coef: 1.0    
    kl_ctrl:              
      horizon: 10000      
      kl_target: 0.1      
      type: fixed         
    kl_estimator_type: k3 
    kl_loss_coef: 0.001   
    lambd: 1.0            
    loss_reduction: token_mean                                      
    max_seq_len: null     
    off_policy_correction:
      geo_mask_high: 1.01 
      geo_mask_low: 0.99  
      outlier_token_is_threshold_high: null                         
      outlier_token_is_threshold_low: null                          
      product_mask_high: 2.0                                        
      product_mask_low: 0.5          
      sequence_mask_metric: null     
      sequence_tis_ratio_clip_high: 5.0                                        
      tis_ratio_type: null           
      token_mask_is_threshold_high: null                                       
      token_mask_is_threshold_low: null                                        
      token_tis_ratio_clip_high: 2.0 
    policy_loss_type: regular        
    sapo:                            
      tau_neg: 1.05                  
      tau_pos: 1.0                   
    temperature: 1.0                 
    tis_imp_ratio_cap: -1.0          
    use_entropy_loss: false          
    use_kl_in_reward: false          
    use_kl_loss: false               
    use_tis: false                   
    value_clip: 0.2                  
    value_head_prefix: value_head    
    zero_variance_filter: false      
  bf16: true                         
  ckpt_interval: 10                  
  ckpt_path: /tmp/ckpts/             
  critic:                            
    fsdp_config:                     
      cpu_offload: false             
      fsdp_size: -1                  
      mixed_precision: null          
      reshard_after_forward: true    
      wrap_policy: {}                
    model:                           
      lora:                          
        alpha: 16                    
        dropout: 0.0                 
        exclude_modules: null        
        init_method: kaiming         
        lora_sync_path: /tmp/skyrl_lora_sync                                   
        max_cpu_loras: null          
        max_loras: 1                 
        rank: 0                      
        target_modules: all-linear   
      path: Qwen/Qwen3-4B-Instruct-2507                                        
    model_config_kwargs: {}          
    optimizer_config:                
      adam_betas:                    
      - 0.9                          
      - 0.999                        
      lr: 5.0e-06                    
      max_grad_norm: 1.0             
      num_warmup_steps: 0            
      offload_after_step: true       
      scheduler: constant_with_warmup
      weight_decay: 0.01             
    sequence_parallel_size: 1        
  critic_mini_batch_size: 256        
  disable_fast_tokenizer: false      
  dump_data_batch: false             
  dump_eval_results: true            
  enable_ray_gpu_monitor: true       
  epochs: 1                          
  eval_batch_size: 1024              
  eval_before_train: true            
  eval_interval: 5                   
  export_path: /tmp/exports/         
  flash_attn: true                   
  fully_async:                       
    max_staleness_steps: 4           
    num_parallel_generation_workers: 768                                       
  gradient_checkpointing: true       
  gradient_checkpointing_use_reentrant: false                                  
  hf_save_interval: -1               
  log_example_interval: 1            
  log_path: /tmp/skyrl-logs          
  logger: wandb                      
  logprobs_chunk_size: 1024          
  max_ckpts_to_keep: -1              
  max_prompt_length: 512             
  max_tokens_per_microbatch: 96000   
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: 1  
  placement:                         
    colocate_all: false              
    colocate_policy_ref: true        
    critic_num_gpus_per_node: 1      
    critic_num_nodes: 1              
    policy_num_gpus_per_node: 2      
    policy_num_nodes: 1              
    ref_num_gpus_per_node: 1         
    ref_num_nodes: 1                 
  policy:                       
    fsdp_config:                     
      cpu_offload: false             
      fsdp_size: -1                  
      mixed_precision: null          
      reshard_after_forward: true    
      wrap_policy: {}                
    inference_only_init: false       
    language_model_only: false       
    megatron_config:                 
      context_parallel_size: 1       
      ddp_config:                    
        average_in_collective: true  
        grad_reduce_in_fp32: true    
        overlap_grad_reduce: false   
        overlap_param_gather: false  
      dist_ckpt_optim_fully_reshardable: false                                 
      empty_cuda_cache: true         
      expert_model_parallel_size: 1  
      expert_tensor_parallel_size: null                                        
      freeze_moe_router: false       
      lora_config:                   
        lora_type: lora              
        merge_lora: true             
      model_config_kwargs: {}        
      moe_aux_loss_coeff: 0.0        
      moe_enable_routing_replay: false                                         
      moe_grouped_gemm: true         
      moe_per_layer_logging: false   
      moe_router_dtype: fp32         
      moe_router_enable_expert_bias: null                                      
      moe_router_load_balancing_type: none                                     
      moe_router_score_function: null
      moe_token_dispatcher_type: alltoall                                      
      optimizer_config_kwargs:       
        optimizer_cpu_offload: false 
        optimizer_offload_fraction: 0.0                                        
        overlap_cpu_optimizer_d2h_h2d: false                                   
        use_precision_aware_optimizer: false                                   
      pipeline_model_parallel_size: 1
      tensor_model_parallel_size: 1  
      torch_profiler_config:         
        enable: false                
        ranks: []                    
        save_path: null              
      transformer_config_kwargs:     
        gradient_accumulation_fusion: false                                    
        recompute_granularity: full  
        recompute_method: uniform    
        recompute_modules:           
        - core_attn                  
        recompute_num_layers: 1      
    model:                           
      lora:                          
        alpha: 16                    
        dropout: 0.0                 
        exclude_modules: null        
        init_method: kaiming         
        lora_sync_path: /tmp/skyrl_lora_sync                                   
        max_cpu_loras: null          
        max_loras: 1                 
        rank: 0                      
        target_modules: all-linear   
      path: Qwen/Qwen3-4B-Instruct-2507                                        
    model_config_kwargs: {}          
    optimizer_config:                
      adam_betas:                    
      - 0.9                          
      - 0.999                        
      lr: 1.0e-06                    
      max_grad_norm: 1.0             
      num_warmup_steps: 0            
      offload_after_step: true       
      scheduler: constant_with_warmup
      weight_decay: 0.01             
    record_memory: false             
    sequence_parallel_size: 1        
    use_torch_compile: false         
  policy_mini_batch_size: 256        
  project_name: skyrl                
  recompute_old_logprobs_per_minibatch: true                                   
  ref:                               
    fsdp_config:                     
      cpu_offload: false             
      fsdp_size: -1                  
      mixed_precision: null          
      reshard_after_forward: true    
      wrap_policy: {}                
    language_model_only: false       
    megatron_config:                 
      context_parallel_size: 1       
      ddp_config:                    
        average_in_collective: true  
        grad_reduce_in_fp32: true    
        overlap_grad_reduce: false   
        overlap_param_gather: false  
      dist_ckpt_optim_fully_reshardable: false                                 
      empty_cuda_cache: true         
      expert_model_parallel_size: 1  
      expert_tensor_parallel_size: null                                        
      freeze_moe_router: false       
      lora_config:                   
        lora_type: lora              
        merge_lora: true             
      model_config_kwargs: {}        
      moe_aux_loss_coeff: 0.0        
      moe_enable_routing_replay: false                                         
      moe_grouped_gemm: true         
      moe_per_layer_logging: false   
      moe_router_dtype: fp32         
      moe_router_enable_expert_bias: null                                      
      moe_router_load_balancing_type: none                                     
      moe_router_score_function: null
      moe_token_dispatcher_type: alltoall                                      
      optimizer_config_kwargs:       
        optimizer_cpu_offload: false 
        optimizer_offload_fraction: 0.0                                        
        overlap_cpu_optimizer_d2h_h2d: false                                   
        use_precision_aware_optimizer: false                                   
      pipeline_model_parallel_size: 1
      tensor_model_parallel_size: 1  
      torch_profiler_config:         
        enable: false                
        ranks: []                    
        save_path: null              
      transformer_config_kwargs:     
        gradient_accumulation_fusion: false                                    
        recompute_granularity: full  
        recompute_method: uniform    
        recompute_modules:           
        - core_attn                  
        recompute_num_layers: 1      
    model:                           
      lora:                          
        alpha: 16                    
        dropout: 0.0                 
        exclude_modules: null        
        init_method: kaiming         
        lora_sync_path: /tmp/skyrl_lora_sync                                   
        max_cpu_loras: null          
        max_loras: 1                 
        rank: 0                      
        target_modules: all-linear   
      path: Qwen/Qwen3-4B-Instruct-2507                                        
    model_config_kwargs: {}          
    sequence_parallel_size: 1        
  remove_microbatch_padding: true    
  resume_mode: latest                
  resume_path: null                  
  rope_scaling: null                 
  rope_theta: null                   
  run_name: test_run                 
  seed: 42                           
  sequence_parallel_backend: ulysses 
  strategy: fsdp                     
  tags: null                         
  train_batch_size: 1024             
  update_epochs_per_batch: 1         
  update_ref_every_epoch: false      
  use_expandable_segments: false