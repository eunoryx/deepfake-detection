import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Import your model and dataset
from resnet_model import CNNThumbnailModel
from dataset import VideoDataset  # Adjust based on actual dataset class

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('cnn_roc_curve.png')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Real', 'Fake'])
    plt.yticks(tick_marks, ['Real', 'Fake'])
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'confusion_matrix': conf_matrix
    }

def main():
    # Load configuration
    with open('cnn_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = VideoDataset(
        root_dir=config['dataset']['val_dir'],  # Use validation set for testing
        mode='test',
        num_frames=config['dataset']['num_frames'],
        transform=None  # Adjust based on your dataset class
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = CNNThumbnailModel(
        num_classes=2,
        num_frames=config['dataset']['num_frames']
    )
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    model = model.to(device)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print('\nEvaluation Results:')
    print(f'Accuracy: {results["accuracy"]:.4f}')
    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall: {results["recall"]:.4f}')
    print(f'F1 Score: {results["f1"]:.4f}')
    print(f'AUC: {results["auc"]:.4f}')
    print('Confusion Matrix:')
    print(results['confusion_matrix'])
    
    print('\nResults and visualizations saved.')

if __name__ == '__main__':
    main()