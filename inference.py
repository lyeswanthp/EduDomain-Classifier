"""
Inference script for EduClass domain classifier.
Loads trained model and classifies educational content into academic domains.
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import json
from typing import List, Dict, Union
import argparse

from config import (
    MODEL_NAME, DEVICE, DTYPE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES,
    PATHS, INFERENCE_CONFIG
)


class DomainClassifier(nn.Module):
    """Neural network classifier for domain classification."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract embeddings using last token pooling strategy."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]

    return embedding


class EduClassifier:
    """High-level interface for educational content classification."""

    def __init__(self, model_path: str = PATHS["model_checkpoint"],
                 label_path: str = PATHS["label_mappings"]):
        """
        Initialize the classifier.

        Args:
            model_path: Path to trained classifier checkpoint
            label_path: Path to label mappings JSON file
        """
        self.device = DEVICE

        # Load embedding model
        print(f"Loading embedding model: {MODEL_NAME}")
        self.embedding_model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map='auto',
            torch_dtype=DTYPE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.embedding_model.eval()

        # Load classifier
        print(f"Loading classifier from: {model_path}")
        self.classifier = DomainClassifier()
        self.classifier.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()

        # Load label mappings
        with open(label_path, 'r') as f:
            mappings = json.load(f)
            self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            self.label2id = mappings['label2id']

        print(f"Classifier ready on {self.device}")

    def _get_embeddings(self, texts: List[str]) -> Tensor:
        """Generate embeddings for input texts."""
        # Format passages
        passages = [f'Represent this passage\npassage: {text}' for text in texts]

        # Tokenize
        inputs = self.tokenizer(
            passages,
            max_length=INFERENCE_CONFIG["max_length"] - 1,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Add EOS token
        batch_size = inputs['input_ids'].shape[0]
        eos_token_id = torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long)
        attention_val = torch.ones(batch_size, 1, dtype=torch.long)
        inputs['input_ids'] = torch.cat([inputs['input_ids'], eos_token_id], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_val], dim=1)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

        return embeddings

    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Union[Dict, List[Dict]]:
        """
        Classify educational content into academic domains.

        Args:
            texts: Single text or list of texts to classify
            return_probabilities: If True, return probability scores for all classes

        Returns:
            Dictionary or list of dictionaries with predictions
        """
        # Handle single text
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Get embeddings
        embeddings = self._get_embeddings(texts)

        # Get predictions
        with torch.no_grad():
            logits = self.classifier(embeddings)
            probabilities = torch.exp(logits)  # Convert log probabilities to probabilities
            predicted_ids = logits.argmax(dim=-1).cpu().tolist()

        # Format results
        results = []
        for i, pred_id in enumerate(predicted_ids):
            result = {
                'domain': self.id2label[pred_id],
                'confidence': probabilities[i][pred_id].item()
            }

            if return_probabilities:
                result['all_scores'] = {
                    self.id2label[j]: probabilities[i][j].item()
                    for j in range(len(self.id2label))
                }

            results.append(result)

        return results[0] if single_input else results

    def batch_predict(self, texts: List[str], batch_size: int = None,
                     return_probabilities: bool = False) -> List[Dict]:
        """
        Classify texts in batches for efficiency.

        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing (default from config)
            return_probabilities: If True, return probability scores

        Returns:
            List of dictionaries with predictions
        """
        if batch_size is None:
            batch_size = INFERENCE_CONFIG["batch_size"]

        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.predict(batch, return_probabilities=return_probabilities)
            all_results.extend(results if isinstance(results, list) else [results])

        return all_results


def main():
    """CLI interface for inference."""
    parser = argparse.ArgumentParser(description='Classify educational content into academic domains')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='File containing texts (one per line)')
    parser.add_argument('--model', type=str, default=PATHS["model_checkpoint"],
                       help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default=PATHS["label_mappings"],
                       help='Path to label mappings')
    parser.add_argument('--probabilities', action='store_true',
                       help='Return all class probabilities')

    args = parser.parse_args()

    # Initialize classifier
    classifier = EduClassifier(model_path=args.model, label_path=args.labels)

    # Get texts
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either --text or --file must be provided")

    # Predict
    results = classifier.predict(texts, return_probabilities=args.probabilities)

    # Print results
    if not isinstance(results, list):
        results = [results]

    for i, result in enumerate(results):
        print(f"\n--- Text {i+1} ---")
        print(f"Domain: {result['domain']}")
        print(f"Confidence: {result['confidence']:.4f}")

        if args.probabilities:
            print("\nAll scores:")
            sorted_scores = sorted(result['all_scores'].items(),
                                 key=lambda x: x[1], reverse=True)
            for domain, score in sorted_scores[:5]:  # Top 5
                print(f"  {domain}: {score:.4f}")


if __name__ == "__main__":
    main()
