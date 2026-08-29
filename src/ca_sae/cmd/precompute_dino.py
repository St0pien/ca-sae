import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from .precompute_imagenet import ImageNetDataset


def make_collate_fn(processor):
    def collate_fn(batch):
        images, labels = zip(*batch)

        images = processor(
            list(images),
            return_tensors="pt",
        )

        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels

    return collate_fn


def main(
    output_path: str,
    split: str = "train",
    dinov3_variant: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    batch_size: int = 64,
    num_workers: int = 16,
    prefetch_factor: int = 2,
):
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dinov3_processor = AutoImageProcessor.from_pretrained(dinov3_variant)
    dinov3_model = AutoModel.from_pretrained(dinov3_variant)
    dinov3_model = dinov3_model.to(device)
    dinov3_model.eval()

    dataset = ImageNetDataset(None, split)
    num_images = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=make_collate_fn(dinov3_processor),
    )

    embedding_size = dinov3_model.config.hidden_size
    num_register_tokens = dinov3_model.config.num_register_tokens

    # Determine the number of patch tokens from one batch.
    first_batch = next(iter(dataloader))
    first_images, _ = first_batch

    first_images = {key: value.to(device) for key, value in first_images.items()}

    with torch.inference_mode():
        first_outputs = dinov3_model(**first_images)

    first_patch_tokens = first_outputs.last_hidden_state[
        :,
        1 + num_register_tokens :,
        :,
    ]

    num_patches = first_patch_tokens.shape[1]
    num_tokens = num_images * num_patches

    print(f"DINOv3 model: {dinov3_variant}")
    print(f"Number of images: {num_images}")
    print(f"Number of patches/image: {num_patches}")
    print(f"Embedding dimension: {embedding_size}")
    print(f"Total number of tokens: {num_tokens}")

    embeddings_path = Path(path, "embeddings.npy")
    embeddings_map = np.memmap(
        embeddings_path,
        dtype=np.float16,
        mode="w+",
        shape=(num_tokens, embedding_size),
    )

    labels_path = Path(path, "labels.npy")
    labels_map = np.memmap(
        labels_path,
        dtype=np.int64,
        mode="w+",
        shape=(num_tokens,),
    )

    offset = 0

    with torch.inference_mode():
        for batch in tqdm(
            dataloader,
            desc="Precomputing DINOv3 patch embeddings",
        ):
            images, labels = batch

            images = {key: value.to(device) for key, value in images.items()}

            outputs = dinov3_model(**images)

            # DINOv3 token layout:
            # [CLS] [register tokens] [patch tokens]
            #
            # Keep only the final-layer patch tokens.
            patch_tokens = outputs.last_hidden_state[
                :,
                1 + num_register_tokens :,
                :,
            ]

            local_batch_size = patch_tokens.shape[0]

            # [batch_size, num_patches, embedding_size]
            # -> [batch_size * num_patches, embedding_size]
            patch_tokens = patch_tokens.reshape(
                local_batch_size * num_patches,
                embedding_size,
            )

            # Assign the image's class label to every patch token
            # belonging to that image.
            #
            # [batch_size]
            # -> [batch_size, num_patches]
            # -> [batch_size * num_patches]
            patch_labels = labels[:, None].expand(-1, num_patches).reshape(-1)

            num_batch_tokens = patch_tokens.shape[0]

            embeddings_map[offset : offset + num_batch_tokens] = (
                patch_tokens.cpu().numpy().astype(np.float16)
            )

            labels_map[offset : offset + num_batch_tokens] = patch_labels.cpu().numpy()

            offset += num_batch_tokens

    embeddings_map.flush()
    labels_map.flush()

    del embeddings_map
    del labels_map

    config_dict = {
        "dataset_size": num_tokens,
        "num_patches": num_patches,
        "embedding_size": embedding_size,
        "split": split,
        "dinov3_variant": dinov3_variant,
        "embedding_type": "last_layer_patch_tokens",
    }

    json_path = Path(path, "config.json")
    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dinov3-variant",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    args = parser.parse_args()

    main(**vars(args))


if __name__ == "__main__":
    cli()
