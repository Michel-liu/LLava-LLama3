# LLaVA-LLama3: Large Language and Vision Assistant with LLama3

This repository is an unofficial implementation of training LLava with LLama3.

### TODO
We are working hard on following items.

- [x] Support pre-training with LLava 1.5
- [x] Support finetuneing with LLava 1.5
- [x] Support [S2-Wrapper](https://github.com/bfshi/scaling_on_scales)
- [x] Support CLI inference

## Installation
The code is developed and validated with ```python=3.10.14,pytorch=2.1.2,cuda=11.8```. Higher versions might be as well.

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/Michel-liu/LLava-LLama3
cd LLava-LLama3
```

2. Create your own Python environment with [Anaconda](https://www.anaconda.com/download).
```Shell
conda create -n llava_llama3 python=3.10
conda activate llava_llama3
pip install --upgrade pip  # enable PEP 660 support
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation
LLaVA training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

### Pretrained Models
You can download the pretrained [LLama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [CLIP-Vit](https://huggingface.co/openai/clip-vit-large-patch14-336) from their offical codebase and place them in `./checkpoints`, as follows
```
├── Meta-Llama-3-8B-Instruct
└── clip-vit-large-patch14-336
```

### Pretrain Data (Feature Alignment)
You can download the 558K subset of the LAION-CC-SBU dataset with BLIP captions in [official huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) or [our mirror](https://pan.baidu.com/s/1ksJiGw8ZYinRvWOkcE8W0g?pwd=vm43).
After downloading all of them, organize the data as follows in `./playground/pretrain`,
```
├── images
│   ├── 00000
│   └── ...
└── blip_laion_cc_sbu_558k.json
```

### Finetuning Data (Visual Instruction)
You can download the annotation of the final mixture instruction tuning JSON in [official huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) or [our mirror](https://pan.baidu.com/s/1DF02vQPID_ph08ybGjBvrA?pwd=k6gz), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,
```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
├── vg
│   ├── VG_100K
│   └── VG_100K_2
└── llava_v1_5_mix665k_unid.json # rename this
```

## Training
LLaVA-LLama3 is trained on 8 H100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.
### Hyperparameter
1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-LLama3 | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-LLama3 | 128 | 2e-5 | 1 | 2048 | 0 |


### Pretraining
Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](scripts/v1_5/pretrain.sh).
Our pretrained checkpoint can be found [here](https://huggingface.co/Michelliu/LLaVA_LLaMA3_8B_PreTrain).


Options to note:
- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower checkpoints/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--tune_mm_mlp_adapter True`: train the connector.

> Pretraining takes around 2 hours for LLaVA-LLama3 on 8x H800 (80G).

### Finetuning
Fully finetuning script with DeepSpeed ZeRO-3 : [`finetune.sh`](scripts/v1_5/finetune.sh).
Our pretrained checkpoint can be found [here](https://huggingface.co/Michelliu/LLaVA_LLaMA3_8B_FineTune).


LoRA finetuning script with DeepSpeed ZeRO-3: [`finetune_lora.sh`](scripts/v1_5/finetune_lora.sh).
Our pretrained checkpoint can be found [here](https://huggingface.co/Michelliu/LLaVA_LLaMA3_8B_FineTune_LoRA).


Options to note:
- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower checkpoints/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.
- `--image_aspect_ratio pad`: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
- `--group_by_modality_length True`: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.

> Fully and LoRA finetuning take around 6 hours for LLaVA-LLama3 on 8x H800 (80G).

## CLI inference
Chat about images using LLaVA-LLama3 with the CLI.

```bash
python -m llava.serve.cli \
    --model-path results/llava_llama3_v1_5_8b_finetune \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

## License
This codebase is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Acknowledgement
This project is built on the open source repositories [LLava](https://github.com/haotian-liu/LLaVA), [LLama](https://github.com/meta-llama/llama3), [LLaVA-pp](https://github.com/mbzuai-oryx/LLaVA-pp), etc.
Thanks them for their well-organized codes!