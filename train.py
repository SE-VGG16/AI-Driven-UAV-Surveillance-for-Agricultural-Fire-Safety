import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.mobilenetv2 import mobilenet_v2
from torch.utils.data import DataLoader

# Define the model with SSD and MobileNetV2 backbone
def create_model(num_classes):
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280  # MobileNetV2 last conv layer output
    model = SSD(backbone, num_classes=num_classes, head=SSDHead(in_channels=1280, num_classes=num_classes))
    return model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root="/path/to/dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
num_classes = 2  # Fire and smoke
model = create_model(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Save trained model
torch.save(model.state_dict(), "fire_detection_model.pth")

print("Model training complete and saved.")
