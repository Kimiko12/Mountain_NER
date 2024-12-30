"""
This script demonstrates:
1. Using a custom DataPreparation class (from data_preparation.py) to build a combined dataset.
2. Splitting the dataset into train/test sets.
3. Creating a custom NERDataset for token classification with a BERT tokenizer.
4. Preparing DataLoader objects for subsequent model training.
"""

import ast
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from transformers import BertTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Import your DataPreparation module
from data_preparation import DataPreparation


class NERDataset(Dataset):
    """
    A PyTorch Dataset for Named Entity Recognition tasks using BERT-based tokenizers.

    This class:
        - Takes a DataFrame containing 'tokens' and 'labels' columns.
        - Converts string representations of lists into Python lists.
        - Tokenizes each row using a provided BertTokenizerFast.
        - Aligns labels with sub-tokens produced by the tokenizer (word-level to sub-token-level).
        - Returns input_ids, attention_masks, and labels as tensors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'tokens' and 'labels' columns.
        E.g., '["Mount", "Everest", ...]' and '["B_mount", "I_mount", ...]'.
    tokenizer : BertTokenizerFast
        The BERT tokenizer to be used for tokenization.
    max_len : int, optional
        Maximum length of input sequences, by default 128.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizerFast, max_len: int = 128) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_len

        # Build a mapping {label -> index} based on unique labels in the dataset
        self.label2id = self._build_label2id()

        # Buffers for storing encoded data
        self.encoded_tokens = []
        self.encoded_labels = []
        self.attention_masks = []

        # For each row, prepare tokens, labels, and alignment
        for _, row in self.df.iterrows():
            tokens = self._safe_eval(row['tokens'])
            labels = self._safe_eval(row['labels'])

            encoded_dict = self._tokenize_and_align(tokens, labels)
            self.encoded_tokens.append(encoded_dict['input_ids'])
            self.attention_masks.append(encoded_dict['attention_mask'])
            self.encoded_labels.append(encoded_dict['labels'])

    def _build_label2id(self) -> Dict[str, int]:
        """
        Builds a {label: index} mapping by collecting unique labels from the entire dataset.

        Returns
        -------
        Dict[str, int]
            Mapping of label strings to integer indices.
        """
        all_labels = []
        for row_labels in self.df['labels']:
            row_labels_list = self._safe_eval(row_labels)
            all_labels.extend(row_labels_list)

        unique_labels = sorted(list(set(all_labels)))  # e.g., ["B_mount", "I_mount", "O"]
        return {label: idx for idx, label in enumerate(unique_labels)}

    @staticmethod
    def _safe_eval(val_data: Union[str, List[str]]) -> List[str]:
        """
        Safely converts a string representation of a list (e.g. "['B_mount', 'I_mount']")
        into an actual Python list. If it's already a list, return it as is.

        Parameters
        ----------
        val_data : Union[str, List[str]]
            Either a string that can be parsed into a list, or an already-existing list.

        Returns
        -------
        List[str]
            List of labels or tokens.
        """
        if isinstance(val_data, str):
            return ast.literal_eval(val_data)
        return val_data

    def _tokenize_and_align(self, tokens: List[str], labels: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes the input tokens and aligns the labels to the resulting sub-tokens.

        Parameters
        ----------
        tokens : List[str]
            A list of word-level tokens.
        labels : List[str]
            A list of labels corresponding to the tokens.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        assert len(tokens) == len(labels), (
            f"Number of tokens ({len(tokens)}) does not match number of labels ({len(labels)})."
        )

        # (Optional) Truncate if too long for [CLS] and [SEP].
        if len(tokens) > self.max_length - 2:
            tokens = tokens[: (self.max_length - 2)]
            labels = labels[: (self.max_length - 2)]

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )

        # word_ids maps each sub-token to the original token index
        word_ids = encoding.word_ids(batch_index=0)

        # Align labels
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # For [CLS], [SEP], [PAD] tokens
            else:
                label_str = labels[word_id]
                label_id = self.label2id[label_str]
                aligned_labels.append(label_id)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),       # shape [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

    def __len__(self) -> int:
        """
        Returns the number of rows (examples) in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with tensors for a single example.
        clone().detach() is used to avoid potential in-place modification issues.

        Parameters
        ----------
        idx : int
            Index of the example to retrieve.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing 'input_ids', 'attention_mask', 'labels'.
        """
        return {
            'input_ids': self.encoded_tokens[idx].clone().detach(),
            'attention_mask': self.attention_masks[idx].clone().detach(),
            'labels': self.encoded_labels[idx].clone().detach()
        }


def main() -> None:
    """
    Main function to:
      1) Prepare a combined dataset from local CSV and Hugging Face using DataPreparation.
      2) Split the dataset into train and test.
      3) Create instances of NERDataset for training and testing.
      4) Construct DataLoaders with DataCollatorForTokenClassification.
      5) Print sample batch data for verification.
    """
    # 1. Initialize DataPreparation (example)
    prepare_dataset = DataPreparation(
        data_path_to_first_source='data/mountain_dataset_with_markup.csv',
        base_url='https://datasets-server.huggingface.co/rows'
    )

    # 2. Load local CSV
    df_local = pd.read_csv(prepare_dataset.data_path_to_first_source)

    # 3. Generate the combined dataset
    df_general = prepare_dataset.generate_dataset(df_local)

    # 4. Train/test split
    train_df, test_df = train_test_split(df_general, test_size=0.2, random_state=42)

    # 5. Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    MAX_LEN = 128
    BATCH_SIZE = 8

    # 6. Create NERDataset instances
    train_dataset = NERDataset(df=train_df, tokenizer=tokenizer, max_len=MAX_LEN)
    test_dataset = NERDataset(df=test_df, tokenizer=tokenizer, max_len=MAX_LEN)

    # 7. DataCollator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 8. DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=4
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=4
    )

    # 9. Check the first batch
    first_batch = next(iter(train_dataloader))
    print("input_ids.shape:", first_batch["input_ids"].shape)  # (batch_size, max_length)
    print("attention_mask.shape:", first_batch["attention_mask"].shape)
    print("labels.shape:", first_batch["labels"].shape)

    # Example: print a sample input_ids and labels from the first example in the batch
    print("Sample input_ids from the first example:", first_batch["input_ids"][0])
    print("Sample labels from the first example:", first_batch["labels"][0])
    # Decode part of the sub-tokens
    print("Decoded tokens example:", tokenizer.convert_ids_to_tokens(first_batch["input_ids"][0][:15]))


if __name__ == '__main__':
    main()
