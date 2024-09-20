import hydra
import numpy as np
import random
import os
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from datasets import disable_caching  
from functools import partial  
from torch.utils.data import Subset
from transformers import (
    SamProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging

from src.corpora import get_dataset_dict
from src.metrics import compute_metrics
from src.modeling import SamBaseline, SeqSam


logger = logging.get_logger()
logging.set_verbosity_info()
disable_caching()

MEDSAM = "wanglab/medsam-vit-base"
SAM = "facebook/sam-vit-base"

SLIP_PATH = "data/lidc_slip"
MCL_PATH = "data/lidc_mcl"
AR_PATH = "data/lidc_ar_unet"
AR_LONG_PATH = "data/lidc_ar_long"


SEED = 42


def set_seed(seed: int = None):
    # Set random seed
    if seed is None:
        seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _main(cfg):
    # Load dataset
    processor = SamProcessor.from_pretrained(cfg.model.load_path)

    # Load model
    if cfg.model.type == "det":
        model = SamBaseline.from_pretrained(
            cfg.model.load_path,
            multimask_output=False
        )

    elif cfg.model.type == "mcl":
        model = SamBaseline.from_pretrained(
            cfg.model.load_path,
            multimask_output=True
        )

    elif cfg.model.type == "seqsam":
        model = SeqSam.from_pretrained(
            cfg.model.load_path,
            num_samples=cfg.model.num_samples,
            ablation=cfg.model.ablation
        )

    else:
        raise ValueError(f"Model type {cfg.model.type} not supported")

    dataset = get_dataset_dict(processor, cfg)
    
    # Downsample the training set to 100 samples for debugging
    def _filter(dataset):
        num_samples = min(len(dataset), 100)
        return Subset(dataset, list(range(num_samples)))
    
    if cfg.debug:
        dataset["train"] = _filter(dataset["train"])
        dataset["eval"] = _filter(dataset["eval"])

    # Print number of parameters
    logger.info(f"Number of parameters: {model.num_parameters()}")
 
    # Make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("sam.vision_encoder"):
            param.requires_grad_(False)
    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=cfg.model.save_path,
        num_train_epochs=cfg.params.num_train_epochs,
        per_device_train_batch_size=cfg.params.batch_size,
        per_device_eval_batch_size=cfg.params.batch_size,
        dataloader_drop_last=False,
        weight_decay=cfg.params.weight_decay,
        logging_dir="./logs",
        logging_steps=cfg.params.logging_steps,
        evaluation_strategy="epoch",
        #evaluation_strategy="steps",
        #eval_steps=200,
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=cfg.params.learning_rate,
        report_to="none"
    )

    r_path = "_".join(
        [cfg.model.type, cfg.model.ablation, cfg.data.dataset]) + ".jsonl"
    _compute_metrics = partial(
        compute_metrics, write_path=os.path.join(cfg.data.path, r_path)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        compute_metrics=_compute_metrics
    )

    if cfg.mode == "train":
        trainer.evaluate()
        trainer.train()

        model.save_pretrained(cfg.model.save_path)
        processor.save_pretrained(cfg.model.save_path)

    results = trainer.evaluate()
    logger.info(results)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    set_seed(cfg.params.seed)
    _main(cfg)


if __name__ == "__main__":
    main()
