import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.mobilenetv2 import mobilenet_v2

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

# Define preprocessing for test dataset
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load test dataset
def load_test_dataset(dataset_path, batch_size=16):
    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

test_dataset_path = "/path/to/test/dataset"
test_loader = load_test_dataset(test_dataset_path)

# Run inference
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        print(f"Predictions: {outputs}")

print("Testing complete.")
