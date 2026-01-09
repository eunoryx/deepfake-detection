import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F

class FrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=4, transform=None):
        self.samples = []
        self.num_frames = num_frames
        self.transform = transform or transforms.ToTensor()
        
        # Initialize face detector
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            self._download_haarcascade()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load dataset
        self._load_dataset(root_dir)

    def _download_haarcascade(self):
        """Download haarcascade if not found"""
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        print("Downloading haarcascade...")
        urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')

    def _load_dataset(self, root_dir):
        """Load dataset from directory structure"""
        for label in ['real', 'fake']:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for video_folder in os.listdir(label_dir):
                video_path = os.path.join(label_dir, video_folder)
                if not os.path.isdir(video_path):
                    continue

                frame_files = sorted([f for f in os.listdir(video_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                if len(frame_files) >= self.num_frames:
                    self.samples.append((video_path, frame_files, 0 if label == 'real' else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_files, label = self.samples[idx]
        
        # Randomly sample frames
        sampled_frames = random.sample(frame_files, self.num_frames) if len(frame_files) > self.num_frames else frame_files
        
        frames = []
        for frame_file in sampled_frames:
            frame_path = os.path.join(video_path, frame_file)
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        frames_tensor = torch.stack(frames)
        thumbnail = self._generate_thumbnail(frames_tensor)  # Corrected method name
        return thumbnail, label

    def _generate_thumbnail(self, frames):
        """Generate 2x2 grid with ONE random square mask per face"""
        T, C, H, W = frames.shape
        masked_frames = []
        
        for t in range(T):
            # Convert frame to numpy
            frame_np = (frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            mask = torch.ones(H, W)
            
            # Detect face
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                
                # Generate ONE random square mask (20-30% of face width)
                mask_size = random.randint(int(w*0.2), int(w*0.3))
                mx = x + random.randint(0, w - mask_size)
                my = y + random.randint(0, h - mask_size)
                
                # Apply single black square
                cv2.rectangle(mask.numpy(), 
                            (mx, my),
                            (mx + mask_size, my + mask_size),
                            0, -1)  # 0 = black, -1 = filled
            
            # Apply mask to all channels
            mask_3channel = mask.unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1)
            masked_frame = frames[t] * mask_3channel
            
            # Downsample
            downsampled = F.interpolate(masked_frame.unsqueeze(0), 
                                      size=(H//2, W//2), 
                                      mode='bilinear').squeeze(0)
            masked_frames.append(downsampled)
        
        # Create 2x2 grid
        rows = [torch.cat(masked_frames[i*2:(i+1)*2], dim=2) for i in range(2)]
        return torch.cat(rows, dim=1)

def visualize_sample(dataset, idx):
    """Visualize sample with proper scaling"""
    thumbnail, label = dataset[idx]
    
    # Convert to numpy and scale to [0,1]
    thumbnail_np = thumbnail.permute(1, 2, 0).numpy()
    thumbnail_np = (thumbnail_np - thumbnail_np.min()) / (thumbnail_np.max() - thumbnail_np.min())
    
    plt.figure(figsize=(6, 6))
    plt.imshow(thumbnail_np)
    plt.title(f"Label: {'Fake' if label else 'Real'}")
    plt.axis('off')
    plt.show()

# Usage
try:
    dataset = FrameDataset(root_dir=r"C:\Users\mohan\Desktop\projects\DIP\archive")
    print(f"Dataset loaded with {len(dataset)} samples")
    visualize_sample(dataset, 0)  # View first sample
except Exception as e:
    print(f"Error: {str(e)}")
    print("Please verify:")
    print("1. Dataset path exists and contains 'real' and 'fake' folders")
    print("2. OpenCV is installed (pip install opencv-python)")
    print("3. Haar cascade file is present")