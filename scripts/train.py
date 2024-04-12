import sys
import os
import gdown
import random
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from datasets import set_caching_enabled    
from dataclasses import dataclass, field
import torch.nn.functional as F
from torch.utils.data import Subset
from transformers import (
    HfArgumentParser, 
    SamProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging
from datasets import Dataset
import numpy as np
import os
import pickle
import torch

from src import constants
from src.corpora import PreprocessingStrategy
from src.metrics import compute_metrics
from src.modeling2 import SamBaseline, SLIP, SamThetaForTraining, UNet, StochasticSam, BlankClassifier, ZcaSam
from src.modeling.probabilistic import ProbabilisticSam
from src.modeling.ssn import SsnSam
from src.utils import set_seed

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/data_lidc.pickle"):
    file_id = "1QAtsh6qUgopFx1LJs20gOO9v5NP6eBgI"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "data/data_lidc.pickle", quiet=False)

set_seed()
logger = logging.get_logger()
logging.set_verbosity_info()
set_caching_enabled(False)
os.environ["WANDB_DISABLED"] = "true"

MEDSAM = "wanglab/medsam-vit-base"
SAM = "facebook/sam-vit-base"

@dataclass
class ModelArguments:
    model_load_path: str = field(
        default=SAM,
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    processor_load_path: str = field(
        default=SAM,
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    teacher_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    model_save_path: str = field(
        default="data/lidc_slip",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    dataset: str = field(
        default="lidc",
        metadata={"help": "Path to the dataset or dataset identifier from huggingface.co/datasets",
                    "choices": ["busi", "cvc", "isic", "lidc"]}
    )

    model_type: str = field(
        default="zca",
        metadata={"help": "Model type", "choices": ["slip", "baseline", "theta", "unet", "stochastic", "classifier", "zca"]}
    )

    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )

    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Number of training epochs"}
    )

    use_bounding_box: bool = field(
        default=True,
        metadata={"help": "Whether to use bounding boxes"}
    )

    use_input_masks: bool = field(
        default=False,
        metadata={"help": "Whether to use bounding boxes"}
    )

    num_simulations: int = field(
        default=10,
        metadata={"help": "Number of simulations for SLIP"}
    )


def get_bounding_box(ground_truth_map, add_perturbation=False):
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

    def __init__(
        self, 
        dataset_location, 
        processor=None, 
        transform=None, 
        use_bounding_box=True, 
        multilabel=True
    ):
        self.transform = transform
        self.processor = processor
        self.use_bounding_box = use_bounding_box
        self.multilabel = multilabel

        self.model = None
        
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
        
        for i, (key, value) in enumerate(data.items()):
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

        self.num_labels = 4

    def __getitem__(self, indices):
        if type(indices) == int:
            index = indices
            image = np.expand_dims(self.images[index], axis=0)
            label = self.labels[index][random.randint(0, self.num_labels-1)].astype(float)

            image = np.repeat(image.transpose(1, 2, 0), 3, axis=2)
            inputs = self.processor(image, do_rescale=False, return_tensors="pt")
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}
            inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)

            inputs["labels"] = F.interpolate(
                torch.tensor(label).unsqueeze(0).unsqueeze(0),  
                size=(256, 256), 
                mode="nearest"
            ).bool().squeeze()
        else:
            bsz = len(indices)
            image = np.stack([self.images[index] for index in indices], axis=0)
            label = np.stack([
                self.labels[index] for index in indices]).astype(float)
            
            # Prepare image and prompt for the model
            if self.processor is not None:
                image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
                input_boxes = None
                if self.use_bounding_box:
                    input_boxes = []
                    for l in label:
                        while True:
                            idx = random.randint(0, 3)
                            if l[idx].sum() > 0:
                                break
                        input_boxes.append([get_bounding_box(l[idx], add_perturbation=False)])  
                
                #input_boxes = [[get_bounding_box(l[random.randint(0, 3)], add_perturbation=True)] for l in label] if self.use_bounding_box else None
                inputs = self.processor(image, input_boxes=input_boxes, do_rescale=False, return_tensors="pt")

                inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device).unsqueeze(0).expand(bsz, -1)
                
                inputs["labels"] = F.interpolate(
                    torch.tensor(label), 
                    size=(256, 256), 
                    mode="nearest"
                ).bool().squeeze(1)

                # Create a mask for labels that are blank
                inputs["label_mask"] = torch.sum(inputs["labels"], dim=(-1, -2)) > 0


            else:
                inputs = {"labels": torch.tensor(label), "pixel_values": torch.from_numpy(image).float()}
        
        return inputs

    def __len__(self):
        return len(self.images)


def _main(args):
    # Load dataset
    processor = SamProcessor.from_pretrained(args.processor_load_path) if args.model_type != "unet" else None

    # Load model
    if args.model_type == "slip":
        model = SLIP.from_pretrained(
            args.model_load_path, 
            processor,
            num_simulations=args.num_simulations,
            cache_dir=constants.CACHE_DIR,
            multiple_annotations=False
        )

    elif args.model_type == "baseline":
        model = SamBaseline.from_pretrained(
            args.model_load_path,
            processor=processor,
            multimask_output=True
        )

    elif args.model_type == "theta":
        model = SamThetaForTraining.from_pretrained(args.model_load_path)
        env = SLIP.from_pretrained(
            args.teacher_load_path,
            processor
        )
        model.set_env(env)

    elif args.model_type == "stochastic":
        model = SsnSam.from_pretrained(
            args.model_load_path,
            processor=processor,
        )

    elif args.model_type == "unet":
        model = UNet()

    elif args.model_type == "classifier":
        model = BlankClassifier.from_pretrained(
            args.model_load_path,
        )

    elif args.model_type == "zca":
        model = ZcaSam.from_pretrained(
            args.model_load_path,
        )

    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    dataset = LIDC_IDRI(dataset_location='data/', processor=processor, use_bounding_box=args.use_bounding_box)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    train_indices, valid_indices, test_indices = indices[2*split:], indices[:split], indices[split:split*2]
    dataset = {"train": Subset(dataset, train_indices), "valid": Subset(dataset, valid_indices)}

    # Downsample the training set to 1000 samples for debugging
    dataset["train"] = Subset(dataset["train"], list(range(100)))

    # Downsample the test set to 100 samples for debugging
    dataset["valid"] = Subset(dataset["valid"], list(range(100)))

    # Print number of parameters
    print(f"Number of parameters: {model.num_parameters()}")
 
    # Make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("sam.vision_encoder"): # or name.startswith("sam.prompt_encoder"):
            param.requires_grad_(False)
    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=args.model_save_path,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        dataloader_drop_last=False,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        eval_steps=200,
        save_strategy="epoch",
        #fp16=True,
        #save_total_limit=1,
        learning_rate=args.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics if args.model_type not in ["theta", "classifier"] else None,
    )

    trainer.evaluate()
    trainer.train()

    if args.model_type == "theta":
        model.env = None

    if args.model_type == "unet":
        torch.save(model.state_dict(), args.model_save_path)
    else:
        model.save_pretrained(args.model_save_path)


def main():
    parser = HfArgumentParser((ModelArguments,))
    args, = parser.parse_args_into_dataclasses()
    _main(args)


if __name__ == "__main__":
    main()
