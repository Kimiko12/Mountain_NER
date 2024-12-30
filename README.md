# Named Entity Recognition on Mountain Dataset with BERT

This repository implements a Named Entity Recognition (NER) task using a fine-tuned BERT model. The NER model identifies specific entities such as mountain names in text using the IOB (Inside-Outside-Beginning) labeling approach:

- **B-Mountain**: Beginning of a mountain entity.
- **I-Mountain**: Inside a mountain entity.
- **O**: Tokens that are not part of an entity.

---

## Features

- Prepares and tokenizes raw text data for NER tasks.
- Fine-tunes a BERT model with class weighting for imbalanced datasets.
- Evaluates the model using accuracy, precision, recall, and F1-score.
- Supports inference to predict mountain entities in new text.
- Flexible configuration and modular structure for extensibility.

---

## Project Structure

```
NER_with_Bert/
├── data/                      # Raw and processed datasets
├── model_checkpoints/         # Directory to save trained models
├── config.py                  # Configuration file for constants
├── dataset.py                 # Custom dataset class for NER
├── data_preparation.py        # Data preparation utility
├── train.py                   # Main training script
├── inference.py               # Inference script
├── utils.py                   # Utility functions
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- GPU (optional for faster training)

### Installation

Clone the repository:

```bash
git clone https://github.com/Kimiko12/Mountain_NER.git
cd Mountain_NER
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Update `config.py` to set hyperparameters and paths:

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42
LEARNING_RATE = 2e-5
MODEL_NAME = 'bert-base-uncased'
PATH_TO_MODEL_CHECKPOINT = 'model_checkpoints/'
```

---

## Dataset

This task uses a combination of two datasets:

1. **[Kaggle Mountain Dataset](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset)**  
2. **[Hugging Face Mountain Dataset](https://huggingface.co/datasets/telord/mountains-ner-dataset)**  

### Prepared Dataset Example:
```text
Sentence: "The Andes mountains are stunning."
Tokens: ["The", "Andes", "mountains", "are", "stunning", "."]
Labels: ["O", "B-Mount", "I-Mount", "O", "O", "O"]
```
---

## Workflow

### Data Preparation

Run `data_preparation.py` to process raw datasets into a tokenized format:

```bash
python data_preparation.py
```

This script converts raw datasets into a DataFrame with tokens and IOB labels.

---

### Model Training

Training metrics will be logged. Example metrics:
```
Eval Loss: 0.206
Accuracy: 97.6%
F1 Score: 88.0%
Precision: 83.1%
Recall: 94.3%
```

---

### Inference

Use the `inference.py` script to test the model on new sentences:

```python
from inference import NERInference

model_path = 'model_checkpoints/'
id2label = {0: 'B_mount', 1: 'I_mount', 2: 'O'}

inference = NERInference(model_path=model_path, id2label=id2label, device='cpu')
result = inference.predict("Mount Everest is in the Himalayas.")
print(result)
```
---

## Future Improvements

1. Extend to support additional entity types (e.g., rivers, landmarks).
2. Experiment with newer models like RoBERTa or DeBERTa.
3. Fine-tune hyperparameters using optimization libraries like Optuna.
4. Apply data augmentation for low-resource scenarios.

