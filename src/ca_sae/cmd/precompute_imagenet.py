import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm


class ImageNetDataset(Dataset):
    def __init__(self, preprocess, split):
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split=split)
        self.len = self.dataset.info.splits[split].num_examples
        self.preprocess = preprocess

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item = self.dataset[index]
        sample, target = item["image"], item["label"]

        if isinstance(sample, Image.Image):
            sample = sample.convert("RGB")
        if self.preprocess:
            sample = self.preprocess(sample)

        return sample, target


def main(
    output_path: str,
    split: str = "train",
    clip_variant: str = "ViT-B/32",
    clip_embedding_size: int = 512,
    batch_size: int = 64,
    num_workers=16,
    prefetch_factor=2,
):
    path = Path(output_path)
    path.mkdir(parents=True)

    clip_model, clip_preprocess = clip.load(clip_variant)

    dataset = ImageNetDataset(clip_preprocess, split)
    num_images = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    embeddings_path = Path(path, "embeddings.npy")
    embeddings_map = np.memmap(
        embeddings_path,
        dtype=np.float16,
        mode="w+",
        shape=(num_images, clip_embedding_size),
    )
    labels_path = Path(path, "labels.npy")
    labels_map = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(num_images,))

    with torch.inference_mode():
        offset = 0
        for batch in tqdm(dataloader, desc="Precomputing embeddings"):
            images, labels = batch
            images = images.to("cuda")
            local_batch_size = images.shape[0]

            clip_embeddings = clip_model.encode_image(images)
            embeddings_map[offset : offset + local_batch_size] = (
                clip_embeddings.cpu().numpy()
            )
            labels_map[offset : offset + local_batch_size] = labels.cpu().numpy()
            offset += local_batch_size

    embeddings_map.flush()
    labels_map.flush()
    del embeddings_map
    del labels_map

    config_dict = {
        "dataset_size": num_images,
        "clip_embedding_size": clip_embedding_size,
        "split": split,
    }
    json_path = Path(path, "config.json")
    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-variant", default="ViT-B/32")
    parser.add_argument("--clip-embedding-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    args = parser.parse_args()

    main(**vars(args))


if __name__ == "__main__":
    cli()
