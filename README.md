# Named Entity Recognition on Mountain Dataset with BERT

This repository implements a Named Entity Recognition (NER) system using a fine-tuned BERT model. The NER model is designed to identify specific entities, such as mountain names, within text using the IOB (Inside-Outside-Beginning) labeling scheme:

- **B-Mountain**: Beginning of a mountain entity.
- **I-Mountain**: Inside a mountain entity.
- **O**: Tokens that are not part of an entity.

---

## Features

- **Data Preparation**: Cleans and tokenizes raw text data tailored for NER tasks.
- **Model Fine-Tuning**: Utilizes a BERT model with class weighting to address imbalanced datasets.
- **Comprehensive Evaluation**: Assesses model performance using metrics like accuracy, precision, recall, and F1-score.
- **Inference Capability**: Enables prediction of mountain entities in new, unseen text.
- **Modular Design**: Offers flexible configuration and a modular structure to facilitate extensibility and customization.

---

## Project Structure

```
NER_with_Bert/
├── data/                      # Raw and processed datasets
├── model_checkpoints/         # Directory to save trained models
├── config.py                  # Configuration file for constants and hyperparameters
├── dataset.py                 # Custom dataset class for NER
├── data_preparation.py        # Data preparation utility script
├── train.py                   # Main training script
├── inference.py               # Inference script for making predictions
├── utils.py                   # Utility functions and helpers
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Getting Started

### Accessing the Dataset and Model Weights

Download the dataset and pre-trained model weights from the following Google Drive link:

[Google Drive - Mountain NER Dataset and Model Weights](https://drive.google.com/drive/folders/1RKO4KOj6-9hctxPfYmcD5cuKw-ksP64u?usp=drive_link)

### Prerequisites

- **Python**: Version 3.9 or higher
- **GPU**: Optional, recommended for accelerated training

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Kimiko12/Mountain_NER.git
   cd Mountain_NER
   ```

2. **Install Dependencies**

   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Configure hyperparameters and file paths by updating the `config.py` file:

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

This project leverages a combination of two datasets:

1. **[Kaggle Mountain Dataset](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset)**
2. **[Hugging Face Mountain Dataset](https://huggingface.co/datasets/telord/mountains-ner-dataset)**

### Example of a Prepared Dataset Entry

```text
Sentence: "The Andes mountains are stunning."
Tokens: ["The", "Andes", "mountains", "are", "stunning", "."]
Labels: ["O", "B-Mountain", "I-Mountain", "O", "O", "O"]
```

---

## Workflow

### 1. Data Preparation

Process raw datasets into a tokenized format suitable for NER tasks.

```bash
python data_preparation.py
```

This script converts raw datasets into a DataFrame containing tokens and their corresponding IOB labels.

### 2. Model Training

Train the BERT-based NER model. Training metrics will be logged during the process.

```bash
python train.py
```

**Example Training Metrics:**

```
Eval Loss: 0.206
Accuracy: 97.6%
F1 Score: 88.0%
Precision: 83.1%
Recall: 94.3%
```

### 3. Inference

Use the trained model to predict mountain entities in new sentences.

```python
from inference import NERInference

model_path = 'model_checkpoints/'
id2label = {0: 'B-Mountain', 1: 'I-Mountain', 2: 'O'}

inference = NERInference(model_path=model_path, id2label=id2label, device='cpu')
result = inference.predict("Mount Everest is in the Himalayas.")
print(result)
```

**Expected Output:**

```python
['Mount Everest', 'Himalayas']
```

---

## Future Improvements

1. **Expand Entity Types**: Incorporate additional entity categories such as rivers, landmarks, and regions.
2. **Explore Advanced Models**: Experiment with newer transformer-based models like RoBERTa or DeBERTa for potential performance gains.
3. **Hyperparameter Optimization**: Utilize optimization libraries like Optuna to fine-tune hyperparameters for improved model performance.
4. **Data Augmentation**: Apply data augmentation techniques to enhance model robustness, especially in low-resource scenarios.

---
