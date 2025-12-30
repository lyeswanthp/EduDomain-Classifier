# EduClass

Production-ready domain classifier for educational content using BMRetriever-7B embeddings and neural classification.

## Overview

Classifies educational textbooks and documents into 18 academic domains with high accuracy using state-of-the-art embeddings from BMRetriever-7B and a lightweight neural classifier.

## Architecture

- **Embedding Model**: BMRetriever-7B (4096-dim dense embeddings)
- **Classifier**: 2-layer FFN (4096 → 2048 → 18 classes)
- **Training**: AdamW optimizer with ReduceLROnPlateau and early stopping
- **Dataset**: MedRAG/textbooks (medical and educational content)

## Installation

```bash
git clone https://github.com/yourusername/educlass.git
cd educlass
pip install -r requirements.txt
```

Requires CUDA-capable GPU for optimal performance.

## Quick Start

### Inference (Classify New Text)

```python
from inference import EduClassifier

# Initialize classifier
classifier = EduClassifier(
    model_path='./domainclassifier.pth',
    label_path='./label_mappings.json'
)

# Classify single text
result = classifier.predict("This chapter covers cellular biology and mitosis...")
print(f"Domain: {result['domain']}, Confidence: {result['confidence']:.2f}")

# Batch classification
texts = ["Text 1...", "Text 2...", "Text 3..."]
results = classifier.batch_predict(texts)
```

### CLI Usage

```bash
# Classify single text
python inference.py --text "Your educational content here"

# Classify from file (one text per line)
python inference.py --file texts.txt --probabilities

# Custom model path
python inference.py --text "Content..." --model ./custom_model.pth
```

## Training Pipeline

### 1. Generate Embeddings

```python
from train import save_embeddings, model, train_dataloader, val_dataloader, test_dataloader

save_embeddings(model, train_dataloader, split='train')
save_embeddings(model, val_dataloader, split='val')
save_embeddings(model, test_dataloader, split='test')
```

### 2. Train Classifier

```bash
python train.py
```

This will:
- Load pre-computed embeddings
- Train the classifier with early stopping
- Save best model to `domainclassifier.pth`
- Export label mappings to `label_mappings.json`

### 3. Evaluate Performance

```bash
# Evaluate on test set
python evaluate.py

# Save detailed results
python evaluate.py --save results.json

# Custom paths
python evaluate.py --embeddings ./embeddings_test.pth --model ./domainclassifier.pth
```

## Project Structure

```
educlass/
├── train.py              # Training pipeline and embedding generation
├── inference.py          # Inference interface (Python API + CLI)
├── evaluate.py           # Model evaluation and metrics
├── config.py             # Centralized configuration
├── requirements.txt      # Python dependencies
├── domainclassifier.pth  # Trained model checkpoint
├── label_mappings.json   # Class ID to domain name mappings
└── embeddings_*.pth      # Pre-computed embeddings (train/val/test)
```

## Configuration

Edit [config.py](config.py) to customize:
- Model paths and hyperparameters
- Training configuration (batch size, learning rate, epochs)
- Data processing settings (max length, train/test split)
- Device and dtype settings

## Model Performance

Evaluated on MedRAG textbooks test set:
- **18 domain classes** (Biology, Chemistry, Physics, Mathematics, etc.)
- **Metrics**: Run `python evaluate.py` for detailed per-class precision/recall/F1

## API Reference

### EduClassifier

```python
classifier = EduClassifier(model_path, label_path)

# Single prediction
result = classifier.predict(text, return_probabilities=False)
# Returns: {'domain': str, 'confidence': float}

# Batch prediction
results = classifier.batch_predict(texts, batch_size=8, return_probabilities=True)
# Returns: [{'domain': str, 'confidence': float, 'all_scores': dict}, ...]
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for GPU acceleration)
- ~14GB disk space for BMRetriever-7B model

## License

MIT
