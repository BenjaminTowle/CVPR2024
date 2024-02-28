import numpy as np
import torch
import os
import pickle
import sys
import random
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path
from transformers import DetrFeatureExtractor, DetrForObjectDetection, Trainer, TrainingArguments
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from src.corpora import FileReader
from torch.utils.data import DataLoader

def get_bounding_box(ground_truth_map: np.array, add_perturbation: bool = False, do_rescale: bool = False):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Return default value
        return [0, 0, 1.0, 1.0]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    if add_perturbation:
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

    if do_rescale:
        x_min /= ground_truth_map.shape[1]
        x_max /= ground_truth_map.shape[1]
        y_min /= ground_truth_map.shape[0]
        y_max /= ground_truth_map.shape[0]

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
        input_boxes = [get_bounding_box(label, do_rescale=True)]
        inputs = self.processor(image, do_rescale=False, return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"] = torch.tensor(label).to(inputs["pixel_values"].device)
        inputs["labels"] = torch.nn.functional.interpolate(inputs["labels"].unsqueeze(0).unsqueeze(0), size=(256, 256), mode="nearest").bool().squeeze()
        inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)
        inputs["reshaped_input_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device)

        inputs["class_labels"] = torch.tensor([0])
        inputs["boxes"] = torch.tensor(input_boxes).float()

        return inputs

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)

from torch.utils.data import Subset


class DetrWrapper(DetrForObjectDetection):
    def forward(self, pixel_values, boxes, class_labels):
        labels = [{"boxes": boxes[i], "class_labels": class_labels[i]} for i in range(len(boxes))]
        outputs = super().forward(pixel_values=pixel_values, labels=labels)
        return outputs

def main():
    processor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50", cache_dir="P:/.hf_cache")
    dataset = LIDC_IDRI(dataset_location = 'data/', processor = processor)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]
    dataset = {"train": Subset(dataset, train_indices), "valid": Subset(dataset, test_indices)}

    # Downsample the training set to 1000 samples for debugging
    dataset["train"] = Subset(dataset["train"], list(range(1000)))

    # Downsample the test set to 100 samples for debugging
    dataset["valid"] = Subset(dataset["valid"], list(range(100)))
    #dataloader = DataLoader(dataset["valid"], batch_size=2, shuffle=True)


    model = DetrWrapper.from_pretrained("facebook/detr-resnet-50", cache_dir="P:/.hf_cache", num_queries=1, ignore_mismatched_sizes=True)
    #model.eval()

    #batch = dataloader.__iter__().__next__()
    #batch["labels"] = [{"boxes": batch["boxes"][i], "class_labels": batch["class_labels"][i]} for i in range(len(batch["boxes"]))]

    #outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])

    #print(outputs)

    # Set up trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        dataloader_drop_last=False,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
    )

    trainer.train()

    model.save_pretrained("data/detr_model")


if __name__ == "__main__":
    main()
