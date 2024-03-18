import sys
import os
import gdown
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
from src.modeling import SamBaseline, SLIP, SamThetaForTraining, UNet, StochasticSam
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

@dataclass
class ModelArguments:
    model_load_path: str = field(
        default="data/checkpoint-final",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    processor_load_path: str = field(
        default="facebook/sam-vit-base",
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
        default="slip",
        metadata={"help": "Model type", "choices": ["slip", "baseline", "theta", "unet", "stochastic"]}
    )

    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )

    num_train_epochs: int = field(
        default=10,
        metadata={"help": "Number of training epochs"}
    )

    use_bounding_box: bool = field(
        default=False,
        metadata={"help": "Whether to use bounding boxes"}
    )

    use_input_masks: bool = field(
        default=False,
        metadata={"help": "Whether to use bounding boxes"}
    )

    num_simulations: int = field(
        default=20,
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
        use_input_masks=True,
        multilabel=True
    ):
        self.transform = transform
        self.processor = processor
        self.use_bounding_box = use_bounding_box
        self.use_input_masks = use_input_masks
        self.multilabel = multilabel

        self.model = None
        if use_input_masks:
            from safetensors.torch import load_file
            file_path = 'data/checkpoint-42500/model.safetensors'
            loaded = load_file(file_path)
            model = UNet()
            model.load_state_dict(loaded)
            self.model = model
        
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
        label = np.stack(self.labels[index]).astype(float)
        
        if self.transform is not None:
            image = self.transform(image)

        # Prepare image and prompt for the model
        if self.processor is not None:
            image = np.repeat(image.transpose(1, 2, 0), 3, axis=2)
            input_boxes = [[get_bounding_box(label[0])]] if self.use_bounding_box else None
            inputs = self.processor(image, input_boxes=input_boxes, do_rescale=False, return_tensors="pt")
            # remove batch dimension which the processor adds by default
            inputs = {k:v.squeeze(0) for k,v in inputs.items()}
            inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)
            #inputs["reshaped_input_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)
            
            inputs["labels"] = F.interpolate(
                torch.tensor(label).unsqueeze(0), 
                size=(256, 256), 
                mode="nearest"
            ).bool().squeeze()
        else:
            inputs = {"labels": torch.tensor(label), "pixel_values": torch.from_numpy(image).float()}

        if self.use_input_masks:
            unet_input = torch.from_numpy(image[..., 0]).view(1, 1, *image.shape[:2]).float()  
            input_masks = self.model(unet_input).pred_masks.squeeze(1)
            inputs["input_masks"] = F.interpolate(
                input_masks, size=(256, 256), 
                mode="bilinear", align_corners=False
            ).squeeze(0)
        
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
            cache_dir=constants.CACHE_DIR
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
        model = StochasticSam.from_pretrained(
            args.model_load_path,
            processor=processor,
            num_simulations=args.num_simulations,
            do_clustering=False,
            num_preds=4
        )

    elif args.model_type == "unet":
        model = UNet()

    else:
        raise ValueError(f"Model type {args.model_type} not supported")

    dataset = LIDC_IDRI(dataset_location='data/', processor=processor, use_bounding_box=args.use_bounding_box, use_input_masks=args.use_input_masks)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]
    dataset = {"train": Subset(dataset, train_indices), "valid": Subset(dataset, test_indices)}

    # Downsample the training set to 1000 samples for debugging
    #dataset["train"] = Subset(dataset["train"], list(range(1000)))

    # Downsample the test set to 100 samples for debugging
    dataset["valid"] = Subset(dataset["valid"], list(range(50)))

    # Print number of parameters
    print(f"Number of parameters: {model.num_parameters()}")
 
    # Make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("sam.vision_encoder") or name.startswith("sam.prompt_encoder"):
            param.requires_grad_(False)

     # Set up trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        dataloader_drop_last=False,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        #save_total_limit=1,
        learning_rate=args.learning_rate,
    )



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["valid"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics if args.model_type != "theta" else None,
    )

    results = trainer.evaluate()
    print(results)


def main():
    parser = HfArgumentParser((ModelArguments,))
    args, = parser.parse_args_into_dataclasses()
    _main(args)


if __name__ == "__main__":
    main()
