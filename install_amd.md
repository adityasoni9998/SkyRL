# Installation Instructions for AMD GPUs

## Pull vLLM apptainer image
```bash
export APPTAINER_CACHEDIR="/work1/grahamneubig/adityabs/apptainer_cache_dir"
mkdir $APPTAINER_CACHEDIR 
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
mkdir $APPTAINER_TMPDIR
apptainer pull /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif docker://docker.io/vllm/vllm-openai-rocm:v0.20.2
```

## Install Dependencies

- Enter apptainer first
```bash
cd /work1/grahamneubig/adityabs/skyrl_max_tokens_per_microbatch
apptainer shell \
    --hostname "$(hostname -s)" \
    --bind "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem,/etc/pki/tls/certs:/etc/pki/tls/certs" \
    /work1/grahamneubig/adityabs/vllm_vllm-openai-rocm_v0.20.2.sif 
```

- Run installation inside apptainer
```bash
cp pyproject.toml pyproject.toml.bak
cp pyproject.other.toml pyproject.toml
python -m venv --system-site-packages /work1/grahamneubig/adityabs/.skyrl_venv/
source /work1/grahamneubig/adityabs/.skyrl_venv/bin/activate
/work1/grahamneubig/adityabs/.skyrl_venv/bin/python -m pip install -e '.[fsdp,tinker]'
/work1/grahamneubig/adityabs/.skyrl_venv/bin/python -m pip install -U ray[all]==2.51.1
/work1/grahamneubig/adityabs/.skyrl_venv/bin/python -m pip install flash-linear-attention[rocm]
/work1/grahamneubig/adityabs/.skyrl_venv/bin/python -m pip install orjson torchdata
cp pyproject.toml.bak pyproject.toml
rm -rf pyproject.toml.bak
```