import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 10    
SEED = 42
LEARNING_RATE = 2e-5 
WEIGHT_DECAY = 0.01
MODEL_NAME = 'bert-base-uncased'
PATH_TO_MODEL_CHECKPOINT = 'model_checkpoints/model.bin'
