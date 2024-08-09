import os
from pathlib import Path
import random
from training.synthesize import load_model
import time
import argparse
import logging
from os.path import dirname, abspath
import sys

logging.getLogger().setLevel(logging.INFO)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from training import DEFAULT_ALPHABET, SEED
from training.clean_text import clean_text
from training.voice_dataset import VoiceDataset
from training.checkpoint import load_checkpoint, save_checkpoint, warm_start_model
from training.validate import validate
from training.utils import (
    get_available_memory,
    get_batch_size,
    get_learning_rate,
    load_labels_file,
    check_early_stopping,
    calc_avgmax_attention,
    train_test_split,
    validate_dataset,
)
from training.tacotron2_model import Tacotron2, TextMelCollate, Tacotron2Loss
from training.tacotron2_model.utils import process_batch
from training.synthesize import text_to_sequence, generate_graph
import time 
import numpy as np
                              
MINIMUM_MEMORY_GB = 4
WEIGHT_DECAY = 1e-6
GRAD_CLIP_THRESH = 1.0
TRAINING_PATH = os.path.join("data", "training")
TENSORBOARD_PATH = os.path.join("data", "tensorboard")
loss_collect = []


def get_mask_from_lengths(lengths, device, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len)).to(device)
    mask = (ids < lengths.to(device).unsqueeze(1)).bool()
    return mask

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous().cuda()
    return torch.autograd.Variable(x)


def get_sizes(data):
    _, input_lengths, _, _, output_lengths = data
    output_length_size = torch.max(output_lengths.data).item()
    input_length_size = torch.max(input_lengths.data).item()
    return input_length_size, output_length_size


def get_y(data):
    _, _, mel_padded, gate_padded, _ = data
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    return mel_padded, gate_padded


def get_x(data):
    text_padded, input_lengths, mel_padded, _, output_lengths = data
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()

    return text_padded, input_lengths, mel_padded, output_lengths


def process_batch(batch, model):
    input_length_size, output_length_size = get_sizes(batch)
    y = get_y(batch)
    y_pred = model(batch, mask_size=output_length_size, alignment_mask_size=input_length_size)

    return y, y_pred

def train(
    audio_directory,
    output_directory,
    metadata_path=None,
    trainlist_path=None,
    vallist_path=None,
    symbols=DEFAULT_ALPHABET,
    checkpoint_path=None,
    transfer_learning_path=None,
    epochs=1,
    batch_size=None,
    early_stopping=True,
    multi_gpu=True,
    iters_per_checkpoint=1,
    iters_per_backup_checkpoint=1,
    train_size=0.8,
    alignment_sentence="",
    logging=logging,
):
    """
    Trains the Tacotron2 model.

    Parameters
    ----------
    audio_directory : str
        Path to dataset clips
    output_directory : str
        Path to save checkpoints to
    metadata_path : str (optional)
        Path to label file
    trainlist_path : str (optional)
        Path to trainlist file
    vallist_path : str (optional)
        Path to vallist file
    symbols : list (optional)
        Valid symbols (default is English)
    checkpoint_path : str (optional)
        Path to a checkpoint to load (default is None)
    transfer_learning_path : str (optional)
        Path to a transfer learning checkpoint to use (default is None)
    epochs : int (optional)
        Number of epochs to run training for (default is 8000)
    batch_size : int (optional)
        Training batch size (calculated automatically if None)
    early_stopping : bool (optional)
        Whether to stop training when loss stops significantly decreasing (default is True)
    multi_gpu : bool (optional)
        Use multiple GPU's in parallel if available (default is True)
    iters_per_checkpoint : int (optional)
        How often temporary checkpoints are saved (number of iterations)
    iters_per_backup_checkpoint : int (optional)
        How often backup checkpoints are saved (number of iterations)
    train_size : float (optional)
        Percentage of samples to use for training (default is 80%/0.8)
    alignment_sentence : str (optional)
        Sentence for alignment graph to analyse performance
    logging : logging (optional)
        Logging object to write logs to

    Raises
    -------
    AssertionError
        If CUDA is not available or there is not enough GPU memory
    RuntimeError
        If the batch size is too high (causing CUDA out of memory)
    """
    assert metadata_path or (
        trainlist_path and vallist_path
    ), "You must give the path to your metadata file or trainlist/vallist files"
    assert torch.cuda.is_available(), "You do not have Torch with CUDA installed. Please check CUDA & Pytorch install"
    os.makedirs(output_directory, exist_ok=True)

    available_memory_gb = get_available_memory()
    assert (
        available_memory_gb >= MINIMUM_MEMORY_GB
    ), f"Required GPU with at least {MINIMUM_MEMORY_GB}GB memory. (only {available_memory_gb}GB available)"

    if not batch_size:
        batch_size = get_batch_size(available_memory_gb)

    learning_rate = get_learning_rate(batch_size)
    logging.info(
        f"Setting batch size to {batch_size}, learning rate to {learning_rate}. ({available_memory_gb}GB GPU memory free)"
    )

    # Set seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)

    # Setup GPU
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # Load model & optimizer
    logging.info("Loading model...")
    model = Tacotron2().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    criterion = Tacotron2Loss()
    logging.info("Loaded model")

    # Load data
    logging.info("Loading data...")
    if metadata_path:
        # metadata.csv
        filepaths_and_text = load_labels_file(metadata_path)
        # random.shuffle(filepaths_and_text)
        # train_files, test_files = train_test_split(filepaths_and_text, train_size)
    # else:
        # trainlist.txt & vallist.txt
        # train_files = load_labels_file(trainlist_path)
        # test_files = load_labels_file(vallist_path)
        # filepaths_and_text = train_files + test_files

    validate_dataset(filepaths_and_text, audio_directory, symbols)
    trainset = VoiceDataset(filepaths_and_text, audio_directory, symbols)
    collate_fn = TextMelCollate()

    # Data loaders
    train_loader = DataLoader(
        trainset, num_workers=0, sampler=None, batch_size=batch_size, pin_memory=False, collate_fn=collate_fn
    )

    logging.info("Loaded data")

    print("train data loader len", len(train_loader))
    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    if checkpoint_path:
        if transfer_learning_path:
            logging.info("Ignoring transfer learning as checkpoint already exists")
        model, optimizer, iteration, epoch_offset = load_checkpoint(checkpoint_path, model, optimizer, train_loader)
        epoch_offset = 0
        iteration += 1
        logging.info("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    elif transfer_learning_path:
        model = warm_start_model(transfer_learning_path, model, symbols)
        logging.info("Loaded transfer learning model '{}'".format(transfer_learning_path))
    else:
        logging.info("Generating first checkpoint...")

    # Enable Multi GPU
    if multi_gpu and torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Alignment sentence
    alignment_sequence = None
    alignment_folder = None
    if alignment_sentence:
        alignment_sequence = text_to_sequence(clean_text(alignment_sentence.strip(), symbols), symbols)
        alignment_folder = os.path.join(TRAINING_PATH, Path(output_directory).stem)
        os.makedirs(alignment_folder, exist_ok=True)

    model.eval()
    validation_losses = []

   
    for _, batch in enumerate(train_loader):

        audio_file = filepaths_and_text[_][0].replace(".wav", "")
        out_file = f"dataset/mel_spectrogram/{audio_file}.npy"

        # if os.path.exists(out_file):
        #     continue

        start = time.perf_counter()
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        # Run without gradients
        with torch.no_grad():
            y, y_pred = process_batch(batch, model)

        np.save(out_file, y_pred[1].detach().cpu().numpy())

        loss = criterion(y_pred, y)
        loss_collect.append((_, loss))

if __name__ == "__main__":
    """Train a tacotron2 model"""
    parser = argparse.ArgumentParser(description="Train a tacotron2 model")
    parser.add_argument("-c", "--checkpoint_path", required=False, type=str, help="checkpoint path")

    args = parser.parse_args()

    train(
        audio_directory="audio",
        output_directory="dataset/checkpoints",
        metadata_path="training/train.txt",
        checkpoint_path=args.checkpoint_path,  
        epochs=1,
        batch_size=1,
        transfer_learning_path=None,
    )

