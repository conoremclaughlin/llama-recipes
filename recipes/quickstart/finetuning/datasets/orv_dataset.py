# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import os
import glob
import json
import copy
import logging
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional


# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(
        images=images, text=text_prompt, padding=True, return_tensors="pt"
    )
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if (
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


class CustomOrvDataset(Dataset):
    """Custom dataset for loading image-text pairs with enhanced validation and prompt templates."""

    SUPPORTED_IMG_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    PROMPT_TEMPLATES = [
        "What do you see in this image?",
        "Please describe this image.",
        "Can you explain what's shown in this image?",
        "What's happening in this image?",
        "Describe the contents of this image.",
    ]

    def __init__(
        self,
        data_dir: str,
        prompt_file: Optional[str] = None,
        validation_ratio: float = 0.1,
        random_prompts: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing image-text pairs
            prompt_file: Optional JSON file containing custom prompts
            validation_ratio: Ratio of data to use for validation
            random_prompts: Whether to randomly select prompts for each sample
        """
        self.data_dir = data_dir
        self.random_prompts = random_prompts

        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Load custom prompts if provided
        self.prompts = self._load_prompts(prompt_file)

        # Get all valid image-text pairs
        self.data_pairs = self._validate_and_load_pairs()

        if not self.data_pairs:
            raise ValueError("No valid image-text pairs found in the directory!")

        self.logger.info(f"Found {len(self.data_pairs)} valid image-text pairs")

    def _load_prompts(self, prompt_file: Optional[str]) -> List[str]:
        """Load custom prompts from JSON file if provided, else use defaults."""
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    custom_prompts = json.load(f)
                if isinstance(custom_prompts, list) and all(
                    isinstance(p, str) for p in custom_prompts
                ):
                    self.logger.info(f"Loaded {len(custom_prompts)} custom prompts")
                    return custom_prompts
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(
                    f"Error loading custom prompts: {e}. Using default prompts."
                )
        return self.PROMPT_TEMPLATES

    def _validate_and_load_pairs(self) -> List[Dict[str, str]]:
        """Validate and load image-text pairs with comprehensive error checking."""
        valid_pairs = []

        # Get all potential image files
        image_files = []
        for ext in self.SUPPORTED_IMG_FORMATS:
            image_files.extend(glob.glob(os.path.join(self.data_dir, f"*{ext}")))

        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            text_path = os.path.join(self.data_dir, f"{base_name}.txt")

            # Validate pair
            if not os.path.exists(text_path):
                self.logger.warning(f"Missing text file for image: {image_path}")
                continue

            try:
                # Validate image
                with Image.open(image_path) as img:
                    img.verify()
                    # Try converting to RGB to catch potential issues
                    img = Image.open(image_path).convert("RGB")

                # Validate text
                with open(text_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                    if not caption:
                        raise ValueError("Empty caption")

                valid_pairs.append(
                    {
                        "image_path": image_path,
                        "text_path": text_path,
                        "caption": caption,
                    }
                )

            except Exception as e:
                self.logger.warning(f"Error validating pair {base_name}: {str(e)}")
                continue

        return valid_pairs

    def _get_prompt(self) -> str:
        """Get a prompt either randomly or sequentially."""
        if self.random_prompts:
            return random.choice(self.prompts)
        return self.prompts[0]

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with error handling."""
        pair = self.data_pairs[idx]

        try:
            # Load and verify image
            image = Image.open(pair["image_path"]).convert("RGB")

            # Format in the expected structure for the DataCollator
            sample = {
                "images": [image],
                "texts": [{"user": self._get_prompt(), "assistant": pair["caption"]}],
            }
            return sample

        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a default sample or raise the error depending on your needs
            raise


def get_custom_dataset(
    dataset_config, processor: any, split: str, split_ratio: float = 0.9
) -> Dataset:
    """
    Get custom dataset with specified split.

    Args:
        dataset_config: Configuration dictionary
        processor: Model processor
        split: Which split to return ("train" or "test")
        split_ratio: Ratio for train/test split

    Returns:
        Appropriate dataset split
    """
    # Get prompt file path from config if provided
    # prompt_file = dataset_config.get("prompt_file", None)
    prompt_file = None
    random_prompts = False  # dataset_config.get("random_prompts", True),

    # Initialize the custom dataset
    full_dataset = CustomOrvDataset(
        data_dir="/workspace/llama-recipes/recipes/quickstart/finetuning/datasets/orv",
        # data_dir="/content/drive/My Drive/training/orv",
        prompt_file=prompt_file,
        random_prompts=random_prompts,
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio)
    test_size = total_size - train_size

    # Create train/test splits
    from torch.utils.data import random_split

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    return train_dataset if split == "train" else test_dataset


class DataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = (
            "right"  # during training, one always uses padding on the right
        )

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            image_list, sample_list = sample["images"], sample["texts"]
            if len(image_list) > 1:
                raise ValueError("Only support one image per sample")
            image = image_list[0].convert("RGB")  # only use the first image
            dialog = []
            for sample_dict in sample_list:
                if not dialog:
                    # only append image to the first sentence
                    dialog += [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": sample_dict["user"].strip()},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": sample_dict["assistant"].strip(),
                                }
                            ],
                        },
                    ]

                else:
                    dialog += [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sample_dict["user"].strip()}
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": sample_dict["assistant"].strip(),
                                }
                            ],
                        },
                    ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs, images, self.processor)


def get_data_collator(processor):
    return DataCollator(processor)
