"""
Evaluation script for EduClass domain classifier.
Computes comprehensive metrics on test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from typing import Dict, Tuple
import argparse

from config import DEVICE, PATHS, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES


class DomainClassifier(nn.Module):
    """Neural network classifier for domain classification."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings."""

    def __init__(self, embeddings: torch.Tensor):
        self.samples = embeddings.shape[0]
        self.embeddings = embeddings[:, 0:-1]
        self.labels = embeddings[:, -1].to(torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

    def __len__(self) -> int:
        return self.samples


def evaluate_model(model: nn.Module, data_loader: DataLoader,
                   device: str = DEVICE) -> Dict[str, np.ndarray]:
    """
    Evaluate model on dataset.

    Args:
        model: Trained classifier
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with predictions and true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Get predictions
            logits = model(embeddings)
            probs = torch.exp(logits)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def compute_metrics(predictions: np.ndarray, labels: np.ndarray,
                   id2label: Dict[int, str]) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: Predicted class indices
        labels: True class indices
        id2label: Mapping from class indices to labels

    Returns:
        Dictionary with evaluation metrics
    """
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(labels, predictions, target_names=target_names,
                                   zero_division=0, output_dict=True)

    return {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report': report
    }


def print_metrics(metrics: Dict, id2label: Dict[int, str]):
    """Print evaluation metrics in readable format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")

    print("\nWeighted Averages (by support):")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")

    print("\nMacro Averages (unweighted):")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

    print("\nPer-Class Metrics:")
    print(f"{'Domain':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*72)

    report = metrics['classification_report']
    for i in range(len(id2label)):
        domain = id2label[i]
        if domain in report:
            stats = report[domain]
            print(f"{domain:<30} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
                  f"{stats['f1-score']:>10.4f} {stats['support']:>10.0f}")

    print("\nConfusion Matrix Shape:", metrics['confusion_matrix'].shape)
    print("(Use --save to export full confusion matrix)")


def save_results(metrics: Dict, id2label: Dict[int, str], output_path: str = "./evaluation_results.json"):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    results = {
        'accuracy': float(metrics['accuracy']),
        'precision_weighted': float(metrics['precision_weighted']),
        'recall_weighted': float(metrics['recall_weighted']),
        'f1_weighted': float(metrics['f1_weighted']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report']
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """CLI interface for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate EduClass domain classifier')
    parser.add_argument('--embeddings', type=str, default=PATHS["test_embeddings"],
                       help='Path to test embeddings')
    parser.add_argument('--model', type=str, default=PATHS["model_checkpoint"],
                       help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default=PATHS["label_mappings"],
                       help='Path to label mappings')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--save', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    print(f"Loading embeddings from: {args.embeddings}")
    embeddings = torch.load(args.embeddings, weights_only=True)

    print(f"Loading model from: {args.model}")
    model = DomainClassifier()
    model.load_state_dict(torch.load(args.model, weights_only=True, map_location=DEVICE))
    model.to(DEVICE)

    print(f"Loading label mappings from: {args.labels}")
    with open(args.labels, 'r') as f:
        mappings = json.load(f)
        id2label = {int(k): v for k, v in mappings['id2label'].items()}

    # Create dataset and loader
    dataset = EmbeddingDataset(embeddings)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nEvaluating on {len(dataset)} samples...")

    # Evaluate
    results = evaluate_model(model, data_loader, device=DEVICE)

    # Compute metrics
    metrics = compute_metrics(results['predictions'], results['labels'], id2label)

    # Print results
    print_metrics(metrics, id2label)

    # Save if requested
    if args.save:
        save_results(metrics, id2label, args.save)
    elif args.save is None:
        # Default save location
        save_results(metrics, id2label)


if __name__ == "__main__":
    main()
