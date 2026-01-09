import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms
from dataset import FrameDataset
from resnet_model import ResNetThumbnailModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



print(f"Using device: {device}")

def print_metrics_for_each_class(y_true, y_pred):
    """Print metrics for each class separately"""
    # For class 0 (real)
    precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=1)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=1)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, zero_division=1)
    
    # For class 1 (fake)
    precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=1)
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=1)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, zero_division=1)
    
    print("\nMetrics for class 0 (real):")
    print(f"Precision: {precision_0:.4f}")
    print(f"Recall: {recall_0:.4f}")
    print(f"F1 Score: {f1_0:.4f}")
    
    print("\nMetrics for class 1 (fake):")
    print(f"Precision: {precision_1:.4f}")
    print(f"Recall: {recall_1:.4f}")
    print(f"F1 Score: {f1_1:.4f}")

# training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=20, consistency_weight=0.5, accumulation_steps=2):
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'consistency_loss': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        # training phase
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_consistency_loss = 0.0
        all_preds = []
        all_labels = []

        #reset gradients
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(inputs)

            #classification loss
            cls_loss = criterion(outputs, labels)

            #consistency loss computed during forward pass
            consistency_loss = model.get_consistency_loss()

            #combined loss
            loss = cls_loss + consistency_weight * consistency_loss
            loss = loss / accumulation_steps  #normalize for gradient accumulation

            #backward pass
            loss.backward()

            #update weights only after several batches or at the end
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            #track metrics
            running_loss += loss.item() * inputs.size(0) * accumulation_steps
            running_cls_loss += cls_loss.item() * inputs.size(0)
            running_consistency_loss += consistency_loss.item() * inputs.size(0)

            #track predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            #free up memory
            if torch.cuda.is_available() and (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        #step learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_cls_loss = running_cls_loss / len(train_loader.dataset)
        epoch_consistency_loss = running_consistency_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)

        #validation phase
        model.eval()
        val_running_loss = 0.0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Only classification loss for validation

                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                val_all_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_all_labels, val_all_preds)

        #save history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_epoch_acc)
        history['consistency_loss'].append(epoch_consistency_loss)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Const: {epoch_consistency_loss:.4f}) | Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')
        print(f'Time: {time.time() - start_time:.2f}s')

        #print class distribution in validation set
        unique_val_labels, val_counts = np.unique(val_all_labels, return_counts=True)
        print(f"Val set class distribution: Class 0: {val_counts[0]}, Class 1: {val_counts[1]}")
        unique_val_preds, val_pred_counts = np.unique(val_all_preds, return_counts=True)
        print(f"Val prediction distribution: Class 0: {val_pred_counts[0]}, Class 1: {val_pred_counts[1]}")
        
        print('-' * 50)

        #save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_resnet_model.pth')
            print(f'Model saved with accuracy: {best_val_acc:.4f}')
    #plot training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['consistency_loss'], label='Consistency Loss')
    plt.title('Consistency Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('resnet_training_history.png')

    return history


def main(root_dir, img_size=224):
    # Hyperparameters
    num_frames = 4
    batch_size = 16
    learning_rate = 0.0001
    num_epochs = 20
    consistency_weight = 0.1
    accumulation_steps = 2
    
    # Create dataset without sample limit
    print("Loading dataset")
    try:
        #use normalization for pretrained models
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
        
        dataset = FrameDataset(
            root_dir=root_dir,
            num_frames=num_frames,
            transform=transform
            # Remove max_samples argument here
        )
        print(f"Loaded {len(dataset)} samples")
        
        #print overall class distribution
        all_dataset_labels = [dataset[i][1] for i in range(len(dataset))]
        unique_labels, counts = np.unique(all_dataset_labels, return_counts=True)
        print(f"Full dataset class distribution: {dict(zip(unique_labels, counts))}")
        
    except Exception as e:
        print(f"Dataset error: {e}")
        return

    
    #split dataset in a stratified manner
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    #verify class distribution in splits
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    unique_train, train_counts = np.unique(train_labels, return_counts=True)
    unique_val, val_counts = np.unique(val_labels, return_counts=True)
    print(f"Train set class distribution: {dict(zip(unique_train, train_counts))}")
    print(f"Val set class distribution: {dict(zip(unique_val, val_counts))}")
    
    #create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize ResNet model
    model = ResNetThumbnailModel(num_classes=2, pretrained=True)
    model = model.float() 
    model = model.to(device)
    
    # Learning rate scheduler
    # DEBUG: Check dataset structure
    print(f"\nDEBUG - Type of dataset: {type(dataset)}")
    print(f"DEBUG - Length of dataset: {len(dataset)}")
    print(f"DEBUG - First sample: {dataset[0] if len(dataset) > 0 else 'Empty'}")

    # Check labels in dataset
    all_labels = []
    for i in range(min(10, len(dataset))):  # Check first 10 samples
        _, label = dataset[i]
        all_labels.append(label)
    print(f"DEBUG - First 10 labels: {all_labels}")
    print(f"DEBUG - Unique labels: {set(all_labels)}")
    class_weights = model.compute_class_weights(dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)



    # Only optimize trainable parameters (not frozen ones)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    #train model
    print("Starting training")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        consistency_weight=consistency_weight,
        accumulation_steps=accumulation_steps
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_resnet_model.pth', weights_only=True))
    
    #final eval
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    #print class distributions
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\nTest set class distribution: {dict(zip(unique_labels, counts))}")

    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
    
    #print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:\n{cm}")
    
    #calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    #calculate metrics for both classes
    print_metrics_for_each_class(all_labels, all_preds)
    
    #standard metrics (for backward compatibility)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=1)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=1)
    
    print('\nFinal Evaluation Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (class 1): {precision:.4f}')
    print(f'Recall (class 1): {recall:.4f}')
    print(f'F1 Score (class 1): {f1:.4f}')
    
    # Save metrics to file
    with open('resnet_results.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision (class 1): {precision:.4f}\n')
        f.write(f'Recall (class 1): {recall:.4f}\n')
        f.write(f'F1 Score (class 1): {f1:.4f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for training')  # Add this line
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        print(f"Error: Path {args.root_dir} does not exist")
        exit(1)
    
    main(root_dir=args.root_dir, img_size=args.img_size)