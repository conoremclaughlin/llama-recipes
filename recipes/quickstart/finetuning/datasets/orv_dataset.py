# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import re
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

    def _get_prompt(self) -> str:
        """Get a prompt either randomly or sequentially."""
        if self.random_prompts:
            return random.choice(self.prompts)
        return self.prompts[0]

    def __len__(self) -> int:
        return len(self.data_pairs)

    # for Facebook's original finetuning examples
    def __getitem_facebook__(self, idx: int) -> Dict:
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

    def _clean_caption(self, caption: str) -> str:
        """Clean captions by removing specific character form patterns and terms."""

        # Specific terms to always remove
        terms_to_remove = [
            "yoo-joonghyuk",
            "kim-dokja",
            "jung-heewon",
            "lee-hyunsung",
            "lee-gilyoung",
            "kim-dokja-degraded-fable",
            "orv-style",
            "manhwa drawn using omniscient reader's viewpoint style",
            "omniscient reader's viewpoint manhwa",
            "drawn in omniscient reader's viewpoint style",
            "Omniscient Reader's Viewpoint Manhwa",
            "kim dokja,",
            "two speech bubbles",
            "korean sound effects",
            "1 thought bubble",
            "1 speech bubble",
            '.,',
        ]

        # Patterns to remove completely (with trailing punctuation)
        removal_patterns = {
            'background': r'background-[a-z0-9-]+(?:[,.\s]+|$)',
            'style': r'style-[a-z0-9-]+(?:[,.\s]+|$)',
            'setting': r'setting-[a-z0-9-]+(?:[,.\s]+|$)',
            'Kim Dokja': r'kim-dokja-[a-z0-9-]+(?:[,.\s]+|$)',
            'Yoo Joonghyuk': r'yoo-joonghyuk-[a-z0-9-]+(?:[,.\s]+|$)',
            'Jung Heewon': r'jung-heewon-[a-z0-9-]+(?:[,.\s]+|$)',
            'Lee Hyunsung': r'lee-hyunsung-[a-z0-9-]+(?:[,.\s]+|$)',
            'Shin Yooseung': r'shin-yooseung-[a-z0-9-]+(?:[,.\s]+|$)',
            'Han Sooyoung': r'han-sooyoung-[a-z0-9-]+(?:[,.\s]+|$)',
            'Lee Gilyoung': r'lee-gilyoung-[a-z0-9-]+(?:[,.\s]+|$)',
            'Yoo Sangah': r'yoo-sangah-[a-z0-9-]+(?:[,.\s]+|$)',
            'Lee Jihye': r'lee-jihye-[a-z0-9-]+(?:[,.\s]+|$)',
            'Kim Namwoon': r'kim-namwoon-[a-z0-9-]+(?:[,.\s]+|$)',
        }

        # Character patterns with their replacements
        character_patterns = {
            # 'Kim Dokja': r'kim-dokja-[a-z-]+\b',
            # 'Yoo Joonghyuk': r'yoo-joonghyuk-[a-z-]+\b',
            # 'Jung Heewon': r'jung-heewon-[a-z-]+\b',
            # 'Lee Hyunsung': r'lee-hyunsung-[a-z-]+\b',
            # 'Shin Yooseung': r'shin-yooseung-[a-z-]+\b',
            # 'Han Sooyoung': r'han-sooyoung-[a-z-]+\b',
            # 'Lee Gilyoung': r'lee-gilyoung-[a-z-]+\b',
            # 'Yoo Sangah': r'yoo-sangah-[a-z-]+\b',
            # 'Lee Jihye': r'lee-jihye-[a-z-]+\b',
            # 'Kim Namwoon': r'kim-namwoon-[a-z-]+\b',
        }

        cleaned_caption = caption

        # Remove specific full terms with their trailing punctuation
        for term in terms_to_remove:
            for punct in [" ", ", ", ". "]:
                cleaned_caption = cleaned_caption.replace(f"{term}{punct}", "")
            if cleaned_caption.endswith(term):
                cleaned_caption = cleaned_caption[: -len(term)]

        # Handle complete removal patterns
        for pattern_name, pattern in removal_patterns.items():
            cleaned_caption = re.sub(pattern, '', cleaned_caption)

        # Replace character form patterns with base names
        for name, pattern in character_patterns.items():
            cleaned_caption = re.sub(pattern, name, cleaned_caption)

        # Clean up base character names if requested
        base_names = [
            "kim dokja",
            "yoo joonghyuk",
            "jung heewon",
            # Add more base names as needed
        ]

        # Optionally remove base names too
        for name in base_names:
            for punct in [" ", ", ", ". "]:
                cleaned_caption = cleaned_caption.replace(f"{name}{punct}", "")
            if cleaned_caption.endswith(name):
                cleaned_caption = cleaned_caption[: -len(name)]

        # Clean up any double spaces and trim
        cleaned_caption = " ".join(cleaned_caption.split())
        # Clean up any trailing punctuation after our removals
        cleaned_caption = re.sub(r'[,.]$', '', cleaned_caption)

        return cleaned_caption.strip()

    def _validate_and_load_pairs(self) -> List[Dict[str, str]]:
        """Validate and load image-text pairs with both standard and who captions."""
        valid_pairs = []

        # Get all potential image files
        image_files = []
        for ext in self.SUPPORTED_IMG_FORMATS:
            image_files.extend(glob.glob(os.path.join(self.data_dir, f"*{ext}")))

        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            text_path = os.path.join(self.data_dir, f"{base_name}.txt")
            who_path = os.path.join(self.data_dir, f"{base_name}.who.txt")

            try:
                # Validate image
                with Image.open(image_path) as img:
                    img.verify()
                    # Try converting to RGB to catch potential issues
                    img = Image.open(image_path).convert("RGB")

                caption = None
                who_caption = None

                # Try to load standard caption
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                        if not caption:
                            self.logger.warning(f"Empty caption file: {text_path}")

                # Try to load who caption
                if os.path.exists(who_path):
                    with open(who_path, 'r', encoding='utf-8') as f:
                        who_caption = f.read().strip()
                        if not who_caption:
                            self.logger.warning(f"Empty who caption file: {who_path}")

                # Require at least one type of caption
                if not caption and not who_caption:
                    self.logger.warning(
                        f"No valid captions found for image: {image_path}"
                    )
                    continue

                valid_pairs.append(
                    {
                        "image_path": image_path,
                        "image_name": base_name,
                        "caption": caption,
                        "who_caption": who_caption,
                    }
                )
                print('image_name: ', base_name)
                print('caption: ', caption)
                print('who_caption: ', who_caption)

            except Exception as e:
                self.logger.warning(f"Error validating pair {base_name}: {str(e)}")
                continue

        return valid_pairs

    def _create_qa_pairs(
        self, image_name: str, caption: Optional[str], who_caption: Optional[str]
    ) -> List[Dict]:
        """Create structured Q&A pairs with both types of captions."""
        qa_pairs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What comic, manhwa, or manga is this?"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the Korean manhwa Omniscient Reader's Viewpoint, written by Sing Shong.",
                    }
                ],
            },
        ]

        # Add who question if we have a who caption
        if who_caption:
            qa_pairs.extend(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Who appears in this image and where are they positioned?",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": who_caption}],
                    },
                ]
            )

        # Add description question if we have a standard caption
        if caption:
            cleaned_caption = self._clean_caption(caption)
            qa_pairs.extend(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Can you describe what's happening in this image?",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": cleaned_caption}],
                    },
                ]
            )

        return qa_pairs

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample with both caption types if available."""
        pair = self.data_pairs[idx]

        try:
            image = Image.open(pair["image_path"]).convert("RGB")

            # Create Q&A pairs using both caption types
            qa_pairs = self._create_qa_pairs(
                pair["image_name"], pair["caption"], pair["who_caption"]
            )

            # Insert image into first question
            qa_pairs[0]["content"].insert(0, {"type": "image", "image": image})

            return {"messages": qa_pairs}

        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {str(e)}")
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

    data_dir = dataset_config.data_dir or "/content/drive/My Drive/training/orv"
    print('data_dir: ', data_dir)
    # Initialize the custom dataset
    # scp -i ~/.ssh/id_inkwell_ed25519 -P 22108 -r ./orv root@194.68.245.29:/workspace/llama-recipes/recipes/quickstart/finetuning/datasets
    full_dataset = CustomOrvDataset(
        data_dir=data_dir,
        # data_dir="/workspace/llama-recipes/recipes/quickstart/finetuning/datasets/orv",
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
