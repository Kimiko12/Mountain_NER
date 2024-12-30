"""
This script provides an NERInference class to perform token classification (NER) inference
using a fine-tuned Transformer model (e.g., BERT). The model and tokenizer are loaded
from a local directory, and predictions are aligned to tokens with offset mappings.
"""

import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Optional


class NERInference:
    """
    A class for Named Entity Recognition (NER) inference using a pre-trained model
    from the Transformers library (Hugging Face).

    Attributes
    ----------
    model_path : str
        The local path to the fine-tuned model.
    id2label : Dict[int, str]
        A mapping from numerical label IDs to label strings (e.g., {0: 'B_mount', 1: 'I_mount', 2: 'O'}).
    device : str
        The device ('cpu' or 'cuda') on which the model should be loaded.

    Methods
    -------
    predict(sentence: str) -> List[Dict[str, str]]:
        Predict token-level labels for the given sentence.
    """

    def __init__(self, model_path: str, id2label: Dict[int, str], device: str = 'cpu') -> None:
        """
        Initialize the NERInference object with a pre-trained model and tokenizer.

        Parameters
        ----------
        model_path : str
            Path to the locally saved transformer model.
        id2label : Dict[int, str]
            Mapping from label IDs to label strings.
        device : str, optional
            Device to load the model onto ('cpu' or 'cuda'), by default 'cpu'.
        """
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True
        )

        self.id2label = id2label
        self.device = device
        self.model.eval()

    def predict(self, sentence: str) -> List[Dict[str, str]]:
        """
        Perform prediction on a single sentence, returning a list of tokens and their labels.

        Parameters
        ----------
        sentence : str
            The input sentence for which to perform NER.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries, each containing {'word': token, 'label': predicted_label}.
        """
        encoded_input = self.tokenizer(
            sentence,
            truncation=True,
            padding=True,
            return_tensors='pt',
            return_offsets_mapping=True
        ).to(self.device)

        # Extract offset mapping before passing the rest of the dictionary to the model
        offset_mapping = encoded_input.pop('offset_mapping')
        with torch.no_grad():
            output = self.model(**encoded_input)
            logits = output.logits  # [batch_size, seq_len, num_labels]

        # Process only the first (and only) item in the batch
        logits = logits[0]
        offset_mapping = offset_mapping[0].tolist()
        word_ids = self._generate_word_ids(offset_mapping)

        # Align the predictions to the word indices
        pred_labels = self._align_predictions(logits, word_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])

        # Build a final list of {word, label} results, skipping special tokens
        results = []
        for token, label, word_id in zip(tokens, pred_labels, word_ids):
            if word_id is not None:
                results.append({'word': token, 'label': label})

        return results

    def _generate_word_ids(self, offset_mapping: List[List[int]]) -> List[Optional[int]]:
        """
        Generates a list of word indices (word_ids) for each token based on the offset mapping.

        Parameters
        ----------
        offset_mapping : List[List[int]]
            A list of [start, end] offsets for each token in the sentence.

        Returns
        -------
        List[Optional[int]]
            A list indicating which word each token belongs to.
            If None, indicates that the token is a special token or invalid offset.
        """
        word_ids = []
        prev_word_span = None

        for i, (start, end) in enumerate(offset_mapping):
            # If start == end, token is likely a special/padding token
            if start == end:
                word_ids.append(None)
            elif prev_word_span is None or start > prev_word_span[1]:
                word_ids.append(i)
            else:
                word_ids.append(prev_word_span_index)

            prev_word_span = (start, end)
            prev_word_span_index = i

        return word_ids

    def _align_predictions(self, logits: torch.Tensor, word_ids: List[Optional[int]]) -> List[str]:
        """
        Aligns the model predictions to word indices.

        Parameters
        ----------
        logits : torch.Tensor
            Logits from the model of shape [seq_len, num_labels].
        word_ids : List[Optional[int]]
            List of word indices (or None for special tokens) for each token.

        Returns
        -------
        List[str]
            A list of predicted labels aligned to each token.
            For tokens that share the same word index, the same predicted label is used.
        """
        probs = softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).tolist()

        pred_labels = []
        prev_word_id = None
        current_label_id = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                # For special tokens like [CLS], [SEP], or sub-tokens of zero-length
                pred_labels.append('O')
            elif word_id != prev_word_id:
                current_label_id = pred_ids[i]
                pred_labels.append(self.id2label[current_label_id])
            else:
                # Continue labeling sub-tokens of the same word
                pred_labels.append(self.id2label[current_label_id])

            prev_word_id = word_id

        return pred_labels


def main() -> None:
    """
    A sample main function showing how to use the NERInference class.
    Adjust model_path and id2label as needed.
    """
    model_path = '/home/nikolay/test_task_quantum/NER_with_Bert/model_checkpoints/model.bin'
    id2label = {0: 'B_mount', 1: 'I_mount', 2: 'O'}

    inference = NERInference(model_path=model_path, id2label=id2label, device='cpu')
    sentence = "Mont Blanc, nestled in the Alps, is the tallest mountain in Western Europe."
    results = inference.predict(sentence)

    print(f"Input sentence:\n{sentence}")
    print(f"Prediction:\n{results}")


if __name__ == '__main__':
    main()
