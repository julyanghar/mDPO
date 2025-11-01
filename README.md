# Multimodal Direct Preference Optimization (mDPO)

[mDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/pdf/2406.11839) (published at EMNLP 2024).

[**🌐 Homepage**](https://feiwang96.github.io/mDPO/) | [**📖 Paper**](https://arxiv.org/pdf/2406.11839) | [**💻 Code**](https://github.com/luka-group/mDPO) | [**🤗 Dataset**](https://huggingface.co/datasets/fwnlp/mDPO-preference-data) 


## Updates
* 🔥 [2024-09-04] Initial release of the [mDPO trainer](mdpo_trainer.py). We are currently working on releasing the code for training and evaluating different models.

## Installation

### Prerequisites
- **Python**: 3.10 or 3.11. Earlier versions do not provide the necessary `typing`
  features used by the training scripts.
- **CUDA Toolkit**: 11.8 or newer, together with a recent NVIDIA driver. Training
  LLaVA-1.5 7B with LoRA requires at least one 24GB GPU (e.g., RTX 4090 or A6000).
- **System packages**: `git`, `ffmpeg`, and `libjpeg-turbo` are required for
  cloning the repository and decoding images inside the preference dataset.

### Create a virtual environment
We recommend installing the project in an isolated environment to prevent
dependency conflicts:

```bash
conda create -n mdpo python=3.10 -y
conda activate mdpo
# or use python -m venv .venv && source .venv/bin/activate
```

### Install Python dependencies
All required Python packages (including `transformers`, `accelerate`, `trl`,
`peft`, and `deepspeed`) are pinned in [`requirements.txt`](requirements.txt).
Install them with:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For the best throughput you can optionally install
[`flash-attn`](https://github.com/Dao-AILab/flash-attention) compiled against your
CUDA toolkit, but it is not mandatory.

### Verify the installation
After the dependencies are installed, run a lightweight compilation check to
ensure all modules import correctly:

```bash
python -m compileall bunny/run_mdpo_bunny.py \
  bunny/data_collator_bunny_phi.py mdpo_trainer.py
```

The command finishes without errors when the environment is set up properly.

## Training

### 1. Prepare preference data
Download or curate a JSON/JSONL preference dataset that follows the schema used by
[fwnlp/mDPO-preference-data](https://huggingface.co/datasets/fwnlp/mDPO-preference-data):

```json
{
  "prompt": "<image> ...",
  "chosen": "assistant response preferred by annotators",
  "rejected": "assistant response that should be discouraged",
  "img_path": "path/to/local/image.jpg"
}
```

All `img_path` entries must point to files that are readable by the training process.

### 2. Configure the run
Edit [`bunny/config.yaml`](bunny/config.yaml) (or create a copy) to point to your
model checkpoint, dataset, and output directory. The file can be supplied with
`--config` when launching training. A quick consistency check can be executed
without downloading the model by running:

```bash
python bunny/run_mdpo_bunny.py --config bunny/config.yaml --validate-only
```

The command prints the resolved paths and exits before touching PyTorch or
Transformers, which makes it safe to run on lightweight environments.

### 3. Launch training
With the configuration in place, start mDPO fine-tuning. For a single GPU setup:

```bash
python bunny/run_mdpo_bunny.py --config bunny/config.yaml
```

For multi-GPU training, use `torchrun` or `accelerate launch` as you normally
would when invoking Hugging Face trainers, for example:

```bash
torchrun --nproc_per_node=4 bunny/run_mdpo_bunny.py --config bunny/config.yaml
```

The script will restore the configuration, instantiate the Bunny/LLaVA model,
and write LoRA checkpoints into the configured `output_dir`.

## Evaluation
TBD

## Citation
Please cite the following paper if you find the repo helpful:
```
@article{wang2024mdpo,
  title={mDPO: Conditional Preference Optimization for Multimodal Large Language Models},
  author={Wang, Fei and Zhou, Wenxuan and Huang, James Y and Xu, Nan and Zhang, Sheng and Poon, Hoifung and Chen, Muhao},
  journal={arXiv preprint arXiv:2406.11839},
  year={2024}
}
```
