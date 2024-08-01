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

### Step 2. Clone repo and install dependencies
Install project dependencies.

```bash
cd ~
git clone https://github.com/StrongResearch/chess-hackathon-2.git
cd ~/chess-hackathon-2
pip install -r requirements.txt
```

Also install the python and Jupyter extensions for VSCode so you can run the notebook `gameplay.ipynb`!

### Step 3. Choose a model
Your options are the Strong Transformer or Transformer by PyTorch.

Copy your chosen model file and the corresponding config YAML file from the `~/chess-hackathon-2/models` to `~/chess-hackathon-2`.

Rename these files to `model.py` and `model_config.yaml` respectively.

### Step 4. Update experiment launch file
Visit Control Plane at https://cp.strongcompute.ai and click on the **Projects** tab. Create a new project and note the
Project ID.

Open the experiment launch file `chessGPT.isc` for editing. Insert your project ID.

```bash
isc_project_id = "<isc-project-id>"
experiment_name = "chessGPT"
gpu_type = "24GB VRAM GPU"
nnodes = 12
output_path = "~/outputs/chessGPT"
dataset_id = "f912e556-e78f-48c7-a8f0-1822aba36714"
compute_mode = "cycle"
command = "source ~/.chess/bin/activate && cd ~/chess-hackathon-2/ && torchrun --nnodes=12 --nproc-per-node=6 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$RANK train.py --model-config /root/chess-hackathon-2/model_config.yaml --save-dir $OUTPUT_PATH"
```

Note we are launchin in compute mode "cycle" just for the purpose of validating that our model will train.

### Step 5. Launch training

Launch your GPT model to train.

```bash
cd ~/chess-hackathon-2
isc train chessGPT.isc
```

## Let's talk Chess AI
The inspiration for this hackathon is the research by Adam Karvonen *Chess-GPT's Internal World Model - A Chess-GPT Linear Emergent World Representation* which demonstrated that an autoregressive GPT model can learn to predict the next character in a PGN string, and is thus able to play chess at a respectable Elo. This is a great resource to understand the background for this hackathon.

https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html

We're going to replicate this work!

### Dataset
We have prepared a dataset that you can use if you like. This dataset has been made available as a Public Dataset on Control Plane and is called **"Leela Chess Zero - Training PGNs Test 60"**. 

The code used to generate this dataset, as well as the PyTorch Dataset class `PGN_HDF_Dataset` that uses this dataset can be found in the `utils/datasets.py` file. Much of the pre-processing is concerned with packaging collections of PGN strings into flat `HDF` files for convenience. This file also contains a defined constant `PGN_CHARS` which is a string containing all the characters necessary to construct a valid PGN.

You are welcome to use any training dataset you prefer, in which case the contents of `utils/datasets.py` can serve as a helpful example to follow. For information about how to import datasets into the Strong Compute ISC, refer to the **Datasets** section of the onboarding docs: https://strong-compute.gitbook.io/strong-compute-developer-docs.

One important thing to bear in mind, however, is that our gameplay code will follow the standard PGN format, where a typical game PGN might look as follows. 

```python
pgn = "1.e4 a6 2.Bc4 a5 3.Qf3 a4 4.Qf7"
```

If your training data does not follow this format then your model may struggle to parse the provided inputs during the game.

### Models
We have pre-packaged 2 models as examples to illustrate architectures suitable for this task which can be found in the `models` directory. These models are called `strong_transformer` and `torch_transformer`. They are both encoder-only transformers with causal masking.

To train one of these models, first copy the corresponding `...transformer.py` file and `...config.yaml` file into the `chess-hackathon-2` main directory, and rename them `model.py` and `model_config.yaml` respectively.

You are welcome to develop any model architecture that satisfies the pre-submission validation check (below).

### Training
We have prepared a training script that you are welcome to use which is saved in the `chess-hackathon-2` main directory called `train.py`. This script implements a number of important functions for distributed training on large clusters such as robust checkpointing and **atomic saving** using the `AtomicDirectory` class from `cycling_utils`.

This training script can be launched using the example experiment launch file `chessGPT.isc`. Once you have your model and config file ready (above), and you have updated the `chessGPT.isc` file with your own ISC Project ID, you can launch your model to train with the following command in your terminal.

```bash
isc train chessGPT.isc
```

### Inference (game play)
To understand how your model will be instantiated and called during gameplay, refer to the `gameplay.ipynb` notebook.

### Important Rules & Submission Spec
#### Important rules
You may develop most any kind of model you like, but your submission must adhere to the following rules. 
 - Your submission must conform to the specification (below),
 - Your model must pass the pre-submission validation check (below) to be admitted into the tournament, 
 - Your model must be trained *entirely from scratch* using the provided compute resources. 
 - You *may not* use pretrained models (includes no transfer learning, fine-tuning, or adaptation modules).
 - You *may not* hard-code any moves (e.g. no opening books).
 - You may install any dependencies you wish for the purpose of *training* but for inference (e.g. game play) your model *must not* require any dependencies other than those included in the `requirements.txt` file for this repo.

#### Submission specification
Your submission must consist of the following set of files only.

```bash
/-team-name
    |- model.py
    |- model_config.yaml
    |- checkpoint.pt
```

**Specification for model_config.yaml**
 - The model_config.yaml file must conform to standard yaml syntax.
 - The model_config.yaml file must contain all necessary arguments for instantiating your model. See below for demonstration of how the model_config.yaml is expected to be used during the tournament.
**Specification for model.py**
 - The model.py file must contain a class description of your model, which must be a PyTorch module called Model.
 - The model must not move any weights to the GPU upon initialization, it will be expected to run entirely on the CPU during the tournament.
 - The model must implement a `score` method. 
 - The `score` method must accept as input the following two positional arguments:
  1. A PGN string representing the current game up to the most recent move, and
  2. A string representing a potential next move.
 - The `score` method must return a `float` value which represents a score for the potential move given the PGN, where higher positive scores always indicate preference for selecting that move.
 - The model *must not* require GPU access to execute the `score` method.
**Specification for checkpoint.pt**
 - The checkpoint.pt file must be able to be loaded with the torch.load function.
 - Your model state dictionary must be able to be obtained from the loaded checkpoint object by calling checkpoint[“model”].
**Pre-submission model validation**
Your model must satisfy the pre-submission validation check to gain admittance into the tournament. You can run the pre-submission validation check 
by first saving your `model.py` and `model_config.yaml` files in the `chess-hackathon-2` directory and then running the following.

```bash
python pre_submission_validation.py
```

If successful, this test will return the following.

```bash
Outputs pass validation tests.
Model passes validation test.
```

If any errors are reported, your model has **failed the test** and must be amended in order to be accepted into the tournament.

