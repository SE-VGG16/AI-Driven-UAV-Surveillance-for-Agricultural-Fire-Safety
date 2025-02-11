import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.mobilenetv2 import mobilenet_v2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define the model with SSD and MobileNetV2 backbone
def create_model(num_classes):
    backbone = mobilenet_v2(pretrained=False).features
    backbone.out_channels = 1280  # MobileNetV2 last conv layer output
    model = SSD(backbone, num_classes=num_classes, head=SSDHead(in_channels=1280, num_classes=num_classes))
    return model

# Load trained model
model_path = "fire_detection_model.pth"
num_classes = 2  # Fire and smoke
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define preprocessing for evaluation dataset
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load evaluation dataset
def load_eval_dataset(dataset_path, batch_size=16):
    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

eval_dataset_path = "/path/to/eval/dataset"
eval_loader = load_eval_dataset(eval_dataset_path)

# Run evaluation
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
