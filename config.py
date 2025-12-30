"""
Configuration file for EduClass domain classifier.
Centralized settings for model, training, and inference.
"""

import torch

# Model Configuration
MODEL_NAME = "BMRetriever/BMRetriever-7B"
EMBEDDING_DIM = 4096
HIDDEN_DIM = 2048
NUM_CLASSES = 18

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Training Configuration
TRAIN_CONFIG = {
    "batch_size": 6,
    "learning_rate": 1e-4,
    "epochs": 30,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0,
}

# Data Configuration
DATA_CONFIG = {
    "dataset_name": "MedRAG/textbooks",
    "max_length": 512,
    "test_size": 0.2,
    "val_size": 0.25,
    "random_state": 42,
    "embedding_batch_size": 4,
}

# File Paths
PATHS = {
    "train_embeddings": "./embeddings_train.pth",
    "val_embeddings": "./embeddings_val.pth",
    "test_embeddings": "./embeddings_test.pth",
    "model_checkpoint": "./domainclassifier.pth",
    "label_mappings": "./label_mappings.json",
}

# Inference Configuration
INFERENCE_CONFIG = {
    "batch_size": 8,
    "max_length": 512,
}
