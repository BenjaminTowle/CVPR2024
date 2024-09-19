import hydra
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from datasets import disable_caching    
from torch.utils.data import Subset
from transformers import (
    SamProcessor,
    Trainer,
    TrainingArguments,
)
from functools import partial
from transformers.utils import logging
import os

from src.corpora import get_dataset_dict
from src.metrics import compute_metrics
from src.modeling import SamBaseline, SeqSam
from src.utils import set_seed


logger = logging.get_logger()
logging.set_verbosity_info()
disable_caching()
os.environ["WANDB_DISABLED"] = "true"

MEDSAM = "wanglab/medsam-vit-base"
SAM = "facebook/sam-vit-base"

SLIP_PATH = "data/lidc_slip"
MCL_PATH = "data/lidc_mcl"
AR_PATH = "data/lidc_ar_unet"
AR_LONG_PATH = "data/lidc_ar_long"


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

    dataset = get_dataset_dict(cfg.data.dataset, processor, cfg.mode)
    
    # Downsample the training set to 1000 samples for debugging
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
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        #evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=cfg.params.learning_rate,
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
        compute_metrics=_compute_metrics,
    )

    if cfg.mode == "eval":
        results = trainer.evaluate()
        logger.info(results)
        exit()

    trainer.evaluate()
    trainer.train()

    model.save_pretrained(cfg.model.save_path)
    processor.save_pretrained(cfg.model.save_path)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    set_seed(cfg.params.seed)
    _main(cfg)


if __name__ == "__main__":
    main()
