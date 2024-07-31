# chess-hackathon-2

Welcome hackers! Let's get you all set up and ready to train!

## Quickstart Guide

### Step 1. Virtual environment 
Create and source a new virtual environment.

```bash
cd ~
python3 -m virtualenv ~/.chess
source ~/.chess/bin/activate
```

### Step 2. Install dependencies
Install project dependencies.

```bash
cd ~/chess-hackathon-2
pip install -r requirements.txt
```

### Step 3. Update experiment launch file
Visit Control Plane at https://cp.strongcompute.ai and click on the **Projects** tab. Create a new project and note the
Project ID.

Open the experiment launch file for editing. Insert your project ID.

```bash
isc_project_id = "<isc-project-id>"
experiment_name = "chessGPT"
gpu_type = "24GB VRAM GPU"
nnodes = 12
output_path = "~/outputs/chessGPT"
dataset_id = "f912e556-e78f-48c7-a8f0-1822aba36714"
compute_mode = "cycle"
command = "source ~/.chess/bin/activate && cd ~/chess-hackathon-sydney/ && torchrun --nnodes=12 --nproc-per-node=6 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$RANK train.py --model-config /root/chess-hackathon-sydney/model_config.yaml --save-dir $OUTPUT_PATH"
```

Note we are launchin in compute mode "cycle" just for the purpose of validating that our model will train.

### Step 4. Launch training

Launch your GPT model to train.

```bash
cd ~/chess-hackathon-2
isc train chessGPT.isc
```

## Repo inventory

### Dataset
**COMING SOON**

### Models
**COMING SOON**

### Training
**COMING SOON**

### Inference (game play)
**COMING SOON**