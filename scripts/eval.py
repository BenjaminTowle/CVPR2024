"""
Run inference on a few samples from the validation set, and save them to a folder for visualization.
"""
import os
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from dataclasses import dataclass, field
from datasets import set_caching_enabled
from functools import partial
from transformers import (
    HfArgumentParser, 
    Trainer, 
    TrainingArguments,
)
from transformers import SamProcessor
from transformers.utils import logging

from src import constants
from src.corpora import PreprocessingStrategy
from src.metrics import compute_metrics
from src.modeling import SamBaseline, SLIP
from src.utils import set_seed

set_caching_enabled(False)

set_seed()
torch.set_grad_enabled(False)
logger = logging.get_logger()
logging.set_verbosity_info()


@dataclass
class Arguments:
    model_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    processor_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the processor or identifier from huggingface.co/models"}
    )

    theta_load_path: str = field(
        default="data/simsr",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    write_path: str = field(
        default="data/results.json",
        metadata={"help": "Path to write results"}
    )

    model_save_path: str = field(
        default="data/cvc_baseline_1",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    dataset: str = field(
        default="cvc",
        metadata={"help": "Path to the dataset or dataset identifier from huggingface.co/datasets",
                    "choices": ["busi", "cvc", "isic"]}
    )

    model_type: str = field(
        default="baseline",
        metadata={"help": "Model type", "choices": ["slip", "baseline"]}
    )

    use_bounding_box: bool = field(
        default=True,
        metadata={"help": "Whether to use bounding boxes"}
    )


def _evaluate(model, dataset, write_path="data/results.json"):
    evaluation_arguments = TrainingArguments(
        output_dir="data/", per_device_eval_batch_size=1
    )

    evaluator = Trainer(
        model=model,
        args=evaluation_arguments,
        eval_dataset=dataset,
        compute_metrics=partial(compute_metrics, write_path=write_path),
    )

    results = evaluator.evaluate()

    return results

def main():
    parser = HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    if args.model_type == "baseline":
        model = SamBaseline.from_pretrained(
            args.model_load_path, processor=processor, multimask_output=True)
    
    else:
        model = SLIP.from_pretrained(
            args.model_load_path, 
            processor=processor, 
            num_simulations=constants.NUM_SIMULATIONS,
            num_preds=constants.NUM_PREDS,
            click_strategy="sampling",
            search_strategy="greedy",
            threshold=constants.THRESHOLD,
            model_path=args.theta_load_path,
            theta_tau=constants.THETA_TAU,
            tau=constants.TAU,
            do_reduce=False,
        )

    preprocessing = PreprocessingStrategy.create(args.dataset)()
    dataset = preprocessing.preprocess(
        processor, valid_size=constants.VALID_SIZE, 
        test_size=constants.TEST_SIZE, use_bounding_box=args.use_bounding_box
    )

    # Evaluate
    results = _evaluate(model, dataset["test"], write_path=args.write_path)
    logger.info(results)


if __name__ == "__main__":
    main()
