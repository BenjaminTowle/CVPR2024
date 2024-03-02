"""
Run inference on a few samples from the validation set, and save them to a folder for visualization.
"""
import numpy as np
import os
import pickle
import random
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from dataclasses import dataclass, field
from datasets import set_caching_enabled, Dataset
from functools import partial
from  torch.utils.data import Subset
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
        default="data/lidc_baseline",
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


from abc import ABC, abstractmethod


class BoundingBoxDetector(ABC):
    @abstractmethod
    def get_bounding_box(self, ground_truth_map, add_perturbation=False, image=None):
        pass

class OracleBoundingBoxDetector(BoundingBoxDetector):
    def get_bounding_box(self, ground_truth_map, add_perturbation=False, image=None):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            # Return default value
            return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        if add_perturbation:
            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox


class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, processor, transform=None):
        self.transform = transform
        self.processor = processor
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0,3)].astype(float)
        if self.transform is not None:
            image = self.transform(image)

        # prepare image and prompt for the model
        image = np.repeat(image.transpose(1, 2, 0), 3, axis=2)
        inputs = self.processor(image, input_boxes=None, do_rescale=False, return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"] = torch.tensor(label).to(inputs["pixel_values"].device)
        inputs["labels"] = torch.nn.functional.interpolate(inputs["labels"].unsqueeze(0).unsqueeze(0), size=(256, 256), mode="nearest").bool().squeeze()
        inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)
        inputs["reshaped_input_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)

        return inputs

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)


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
            search_strategy="mcts",
            threshold=constants.THRESHOLD,
            model_path=args.theta_load_path,
            theta_tau=constants.THETA_TAU,
            tau=constants.TAU,
            do_reduce=False,
        )

    dataset = LIDC_IDRI(dataset_location='data/', processor=processor)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]
    dataset = {"train": Subset(dataset, train_indices), "test": Subset(dataset, test_indices)}
    # Downsample the test set to 100 samples for debugging
    dataset["test"] = Subset(dataset["test"], list(range(100)))

    # Evaluate
    results = _evaluate(model, dataset["test"], write_path=args.write_path)
    logger.info(results)


if __name__ == "__main__":
    main()
