# Installation Instructions for NVIDIA GPUs (H100 tested)

## Install Dependencies

```bash
cd /project/flame/adityabs
uv venv --python 3.12
source .venv/bin/activate
cd SkyRL/
uv sync --active --extra tinker --extra fsdp
```