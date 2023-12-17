# DLA. RawNet2 implementation for audio spoofing.

## Preparations
Make sure you run on Python version 3.8-3.10

1. Clone the repo.
    ```
    git clone https://github.com/karimdzan/Hifi-GAN.git
    ```

1. Prepare requirements.
    ```
    cd ASVSpoof
    pip install -r requirements.txt
    ```

1. Download model checkpoint.
    ```
    gdown https://drive.google.com/file/d/1JjyfmCj9XZG6wgIFEuSZVCfxmncsNjTN/view?usp=sharing
    ```

## Inference

1. You can use [test.py](./test.py) to check your audio or audio from tests for spoofing:
    ```
    python3 test.py
    ```
    Don't forget to add your audio paths and checkpoint path to test.yaml in configs folder.

## Training

1. To train your own model, use [train.py](./train.py):
    ```
    python3 train.py
    ```
   You will need to specify your wandb login key in train.yaml config.

## Artifacts

more checkpoints: https://drive.google.com/drive/folders/1QLam3cpSCELEmrxf2-isU8CUu8vp4mp9?usp=sharing
audios: https://drive.google.com/drive/folders/1LizQFwGcrYa3dr5DIJDovZJj46XsLDfh?usp=sharing