import gdown
import nibabel as nib
import numpy as np
import os
import pickle
import random
import torch
import torch.nn.functional as F
from datasets import Dataset
from os.path import join
from PIL import Image
from torch.utils.data import Subset

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/data_lidc.pickle"):
    file_id = "1QAtsh6qUgopFx1LJs20gOO9v5NP6eBgI"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", "data/data_lidc.pickle", quiet=False)



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


class QUBIQ(Dataset):

    file_reader = "nii.gz"

    def __init__(self, processor, split="train", use_bounding_box=True) -> None:
        self.i = 0
        self.processor = processor
        self.images, self.labels = self.preprocess(split)
        self.use_bounding_box = use_bounding_box

    def read_file(self, path: str):
        image = nib.load(path).get_fdata()

        image += np.abs(image.min())
        image = (256 * (image / image.max())).astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        
        return image
    
    def read_label(self, path: str):
        label = nib.load(path).get_fdata()
        label = np.array(Image.fromarray(label).resize((256, 256)))
        label = label > 0
        return label

    def dfs(self, paths: list, path: str):
        for item in os.listdir(path):
            new_path = join(path, item)
            if "kidney" not in new_path:
                continue

            if item.startswith("case"):
                paths.append(new_path)
            elif os.path.isdir(new_path):
                self.dfs(paths, new_path)
            else:
                continue
        return paths

    def get_images_labels(self, path: str, single_label: bool = False) -> dict:
        image = []
        label = []
        cases = self.dfs([], path)

        for case in cases:
            label_set = []
            for file in os.listdir(case):
                if not file.endswith(self.file_reader):
                    continue
                if file.startswith("image"):
                    image.append(join(case, file))
                else:
                    label_set.append(join(case, file))

            if single_label:
                label.append(label_set[self.i % len(label_set)])
                self.i += 1
            else:
                label.append(label_set)
        
        return {
            "image": image,
            "label": label
        }

    def preprocess(self, split="train"):
        dataset_path = "data/qubiq"

        if split == "train":
            path = join(dataset_path, "training_data_v2")

        else:
            path = join(dataset_path, "validation_data_v2")
            
        dict = self.get_images_labels(path)

        images = [self.read_file(img) for img in dict["image"]]
        labels = [[self.read_label(l) for l in lbl] for lbl in dict["label"]]
            
        return images, labels
        
    def __getitem__(self, indices):
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
                        idx = random.randint(0, len(l)-1)
                        if l[idx].sum() > 0:
                            break
                    input_boxes.append([get_bounding_box(l[idx], add_perturbation=False)])  
            
            inputs = self.processor(image, input_boxes=input_boxes, do_rescale=True, return_tensors="pt")

            inputs["original_sizes"] = torch.tensor([256, 256]).to(inputs["pixel_values"].device).unsqueeze(0).expand(bsz, -1)
            
            inputs["labels"] = F.interpolate(
                torch.tensor(label), 
                size=(256, 256), 
                mode="nearest"
            ).bool().squeeze(1)

            # Create a mask for labels that are blank
            inputs["label_mask"] = torch.sum(inputs["labels"], dim=(-1, -2)) > 0

        return inputs

    def __len__(self):
        return len(self.images)
        



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


def get_dataset_dict(dataset_name, processor, args):
    if dataset_name == "lidc":
        dataset = LIDC_IDRI(dataset_location='data/', processor=processor, use_bounding_box=args.use_bounding_box)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        train_indices, valid_indices, test_indices = indices[2*split:], indices[:split], indices[split:split*2]
        dataset = {"train": Subset(dataset, train_indices), "valid": Subset(dataset, valid_indices), "test": Subset(dataset, test_indices)}
        
        return dataset
    
    elif dataset_name == "qubiq":
        train = QUBIQ(processor, split="train", use_bounding_box=args.use_bounding_box)
        dataset_size = len(train)
        indices = list(range(dataset_size))
        train = Subset(train, indices)

        valid = QUBIQ(processor, split="valid", use_bounding_box=args.use_bounding_box)
        dataset_size = len(valid)
        indices = list(range(dataset_size))
        valid = Subset(valid, indices)
        return {"train": train, "valid": valid}
    
    else:
        raise ValueError(f"Dataset {dataset_name} not found.")
