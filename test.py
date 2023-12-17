import argparse
import json
import os
from pathlib import Path
import sys
import torch
import logging
from tqdm import tqdm
import torchaudio
from src.utils import ROOT_PATH
# from src.utils.object_loading import get_dataloaders
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.metric.utils import compute_eer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "model_best.pth"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to checkpoint", 
        required=True)
    parser.add_argument(
        "--audio", 
        type=str,
        help="Dir with audio", 
        required=True)
    return parser.parse_args()

@hydra.main(version_base=None, config_path="src/configs", config_name="test")
def main(config):
    # args = parse_args()
    logger = logging.getLogger("test")
    OmegaConf.resolve(config)

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    # dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = instantiate(config["arch"])

    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for audio_path in os.listdir(config.audio_dir):
            print(audio_path)                
            audio, sr = torchaudio.load(config.audio_dir + '/' + audio_path)
            audio = audio.to(device)
            if 'elon' in audio_path:
                logits = model(audio.unsqueeze(1))
            else:
                logits = model(audio.unsqueeze(0))
            bonafide_prob = torch.softmax(logits[0], 0)
            logger.info("{} {:.6f}".format(audio_path, bonafide_prob[1]))


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()