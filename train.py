"""
Script for fine-tuning a BERT model for token classification (Named Entity Recognition).
The process includes:
1. Data preparation using a custom DataPreparation class.
2. Splitting the dataset into train/test.
3. Creating custom NERDataset objects.
4. Computing class weights to handle label imbalance.
5. Training with a custom Trainer class (CustomTrainer) for a custom loss function.
6. Saving and evaluating the final trained model.
"""

import os
import torch
import logging
import pandas as pd
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from transformers import (
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    get_scheduler
)
from sklearn.model_selection import train_test_split

# Local imports (your custom modules)
from config import MODEL_NAME, DEVICE, MAX_LEN, BATCH_SIZE, EPOCHS, SEED, PATH_TO_MODEL_CHECKPOINT
from dataset import NERDataset
from data_preparation import DataPreparation
from utils import compute_metrics

logging.basicConfig(level=logging.INFO)


def compute_class_weights(dataset: NERDataset, num_labels: int) -> torch.Tensor:
    """
    Computes class weights based on label frequency in the dataset.

    Parameters
    ----------
    dataset : NERDataset
        The dataset from which to compute label frequencies.
    num_labels : int
        The number of unique labels.

    Returns
    -------
    torch.Tensor
        A tensor containing the computed class weights.
    """
    label_counts = [0] * num_labels
    total_count = 0

    for idx in range(len(dataset)):
        labels = dataset[idx]['labels']
        if torch.is_tensor(labels):
            labels = labels.tolist()

        for lbl in labels:
            # Ignore special tokens
            if lbl != -100:
                label_counts[lbl] += 1
                total_count += 1

    # Ensure no zero division
    label_counts = [max(count, 1) for count in label_counts]

    # Compute class weights as ratio of total_count to label_counts
    class_weights = [total_count / count for count in label_counts]
    logging.info(f"Class counts: {label_counts}")
    logging.info(f"Class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)


class CustomTrainer(Trainer):
    """
    A custom Trainer class that allows using class weights when computing the loss.
    """

    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs) -> None:
        """
        Initializes the CustomTrainer.

        Parameters
        ----------
        class_weights : torch.Tensor, optional
            Tensor containing class weights, by default None.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model: BertForTokenClassification, inputs: Dict[str, Any], 
                     return_outputs: bool = False, **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Custom loss function that applies class weights for imbalanced classification.

        Parameters
        ----------
        model : BertForTokenClassification
            The model to compute the loss for.
        inputs : Dict[str, Any]
            Dictionary of inputs: 'input_ids', 'attention_mask', 'labels', etc.
        return_outputs : bool, optional
            Whether to return the model outputs, by default False.

        Returns
        -------
        Tuple[torch.Tensor, Any]
            A tuple containing the loss (and optionally the outputs if return_outputs=True).
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Choose a CrossEntropyLoss with optional class weighting
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train(
    train_dataset: NERDataset,
    test_dataset: NERDataset,
    label2id: Dict[str, int],
    path_to_model_checkpoint: str
) -> BertForTokenClassification:
    """
    Trains a BERT model for token classification, using a custom trainer with class weighting.

    Parameters
    ----------
    train_dataset : NERDataset
        The training dataset.
    test_dataset : NERDataset
        The testing/validation dataset.
    label2id : Dict[str, int]
        Mapping from label strings to label IDs.
    path_to_model_checkpoint : str
        The directory path where the trained model will be saved.

    Returns
    -------
    BertForTokenClassification
        The fine-tuned model.
    """
    num_labels = len(label2id)
    id2label = {v: k for k, v in label2id.items()}

    # Load a pre-trained BERT model for token classification
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        attention_probs_dropout_prob=0.3,
        hidden_dropout_prob=0.3
    ).to(DEVICE)

    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset, num_labels=num_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0
    )

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    total_steps = len(train_dataset) // BATCH_SIZE * EPOCHS
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Create a tokenizer for data collation
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # Initialize the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        optimizers=(optimizer, lr_scheduler)
    )

    # Add an early stopping callback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Start training
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(path_to_model_checkpoint)
    model.save_pretrained(path_to_model_checkpoint)

    # Evaluate and log final metrics
    metrics = trainer.evaluate()
    logging.info(f"Final evaluation metrics: {metrics}")

    return model


def main() -> None:
    """
    Main function to orchestrate training.
    Steps:
    1. Prepare dataset from local CSV and Hugging Face.
    2. Split into train and test sets.
    3. Create NERDataset instances.
    4. Train the model.
    """
    # Initialize data preparation
    prepare_dataset = DataPreparation(
        data_path_to_first_source='data/mountain_dataset_with_markup.csv',
        base_url='https://datasets-server.huggingface.co/rows'
    )

    # Load local data and generate a combined dataset
    df_local = pd.read_csv(prepare_dataset.data_path_to_first_source)
    df_general = prepare_dataset.generate_dataset(df_local)

    # Train/test split
    train_df, test_df = train_test_split(df_general, test_size=0.2, random_state=42)

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # Create dataset objects
    train_dataset = NERDataset(df=train_df, tokenizer=tokenizer, max_len=MAX_LEN)
    test_dataset = NERDataset(df=test_df, tokenizer=tokenizer, max_len=MAX_LEN)

    label2id = train_dataset.label2id

    # Start training
    model = train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        label2id=label2id,
        path_to_model_checkpoint=PATH_TO_MODEL_CHECKPOINT
    )

    logging.info("Training complete.")


if __name__ == '__main__':
    main()
