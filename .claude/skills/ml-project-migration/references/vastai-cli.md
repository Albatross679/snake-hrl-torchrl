# Vast.ai CLI Reference

## Authentication

```bash
pip install vastai
vastai set api-key <API_KEY>
```

API key stored at `~/.config/vastai/vast_api_key`. Also set `VASTAI_API_KEY` env var in `~/.bashrc`.

## Instance Lifecycle

### 1. Search Offers

```bash
# Single GPU, reliable, on-demand
vastai search offers 'reliability>0.98 num_gpus=1 gpu_name=RTX_4090 rented=False'

# Multi-GPU datacenter
vastai search offers 'reliability>0.99 num_gpus=4 gpu_name=RTX_A6000 datacenter=True'

# Budget: sort by price
vastai search offers 'num_gpus=1 gpu_ram>20 reliability>0.95' -o 'dph'

# Region filter
vastai search offers 'num_gpus=1 gpu_name=RTX_4090 geolocation in [US,CA,GB]'

# Minimum specs
vastai search offers 'disk_space>100 duration>24 gpu_ram>20 cuda_vers>=12.1 cpu_ram>32'
```

Key search fields:

| Field | Type | Use |
|-------|------|-----|
| `gpu_name` | string | GPU model (replace spaces with `_`) |
| `num_gpus` | int | GPU count |
| `gpu_ram` | float | Per-GPU VRAM (GB) |
| `cpu_ram` | float | System RAM (GB) |
| `disk_space` | float | Storage (GB) |
| `dph` | float | $/hour |
| `reliability` | float | 0–1 score |
| `cuda_vers` | float | Max CUDA version |
| `datacenter` | bool | Datacenter only |
| `geolocation` | string | 2-letter country code |
| `duration` | float | Max rental days |
| `inet_down` | float | Download Mb/s |

### 2. Launch Instance

Two approaches:

```bash
# Quick launch (auto-selects best offer)
vastai launch instance -g RTX_4090 -n 1 -i vastai/pytorch --ssh --direct -d 64

# Manual: pick offer ID from search, then create
vastai create instance <OFFER_ID> --image vastai/pytorch --ssh --direct --disk 64
```

Common flags:
- `--ssh` — SSH access
- `--direct` — direct (faster) connections
- `--disk N` — disk size in GB
- `--image NAME` — Docker image
- `--label NAME` — human label
- `--onstart-cmd 'CMD'` — startup script
- `--env '-e KEY=VAL -p HOST:CONTAINER'` — env vars and port mappings

### 3. Connect

```bash
# Get SSH command
vastai ssh-url <INSTANCE_ID>
# Returns: ssh -p <port> root@<host> -L 8080:localhost:8080

# Get SCP prefix
vastai scp-url <INSTANCE_ID>
# Returns: scp -P <port> root@<host>:
```

### 4. Manage

```bash
vastai show instances                    # list all instances
vastai show instances -q                 # IDs only
vastai stop instance <ID>                # stop (preserves data, stops billing compute)
vastai start instance <ID>               # restart stopped instance
vastai reboot instance <ID>              # stop + start
vastai destroy instance <ID>             # permanent delete
vastai logs <ID>                         # view logs
vastai label instance <ID> --label 'X'   # rename
```

### 5. Environment Variables (Account-Level)

```bash
vastai show env-vars                     # list
vastai show env-vars -s                  # list with values
vastai create env-var -k KEY -v VALUE    # create
vastai update env-var -k KEY -v VALUE    # update
vastai delete env-var -k KEY             # delete
```

Account-level env vars are injected into every new instance automatically. Use for:
- `WANDB_API_KEY`, `HF_TOKEN`, `GH_TOKEN`, `ANTHROPIC_API_KEY`
- `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`

### 6. File Transfer

```bash
# SCP files to instance
SCP_URL=$(vastai scp-url <ID>)
scp -P <port> .env ${SCP_URL}/workspace/.env

# vastai copy (between instances or local)
vastai copy <SRC_ID>:/path /local/path
vastai copy /local/path <DST_ID>:/path

# Cloud copy (S3/GCS/B2)
vastai cloud copy s3://bucket/path <INSTANCE_ID>:/path
```

## GPU Selection Guide for ML

| Task | Recommended GPU | VRAM | Notes |
|------|----------------|------|-------|
| RL training (single env) | RTX 3090 / 4090 | 24 GB | Best price/perf |
| RL training (vectorized) | RTX A5000 / A6000 | 24–48 GB | More VRAM for batched envs |
| Fine-tuning 7B LLM | RTX 4090 / A100 | 24–80 GB | QLoRA fits on 24 GB |
| Fine-tuning 70B LLM | 4× A100 80GB | 320 GB | Need multi-GPU |
| Inference / eval | RTX 3090 / L4 | 24 GB | Budget-friendly |
| Physics sim (MuJoCo etc.) | RTX 3090 | 24 GB | CPU-bound, GPU for rendering |

## Spot (Interruptible) vs On-Demand

| | Spot | On-Demand |
|---|---|---|
| Price | 30–70% cheaper | Full price |
| Stability | Can be preempted | Guaranteed |
| Use for | Short experiments, sweeps | Long training, production |
| Flag | `--bid_price 0.10` | (default) |
| Search | `vastai search offers -b` | `vastai search offers -d` |

Spot is good for hyperparameter sweeps where interruption = restart one run, not the whole sweep.
