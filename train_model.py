import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 1. SETUP: Define your transformations (resize images to be standard)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize all images to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. DATA: Load your database of images
# Structure your folders like this:
# dataset/
#   train/
#     calculus/
#     ulcer/
#     discoloration/
train_dataset = datasets.ImageFolder('dataset_split/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. MODEL: Create the AI (Using a pre-built brain called ResNet18)
# We use "Transfer Learning" (taking a smart AI and teaching it about teeth)
model = models.resnet18(pretrained=True)

# Change the final layer to output only OUR 3 classes (Calculus, Ulcer, Discoloration)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3) 

# 4. TRAINING LOOP (The learning process)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Start Training the Detective...")

# Loop over the dataset 5 times (5 epochs)
for epoch in range(5): 
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)      # The AI guesses
        loss = criterion(outputs, labels) # Calculate how wrong it was
        loss.backward()              # Learn from mistakes
        optimizer.step()             # Update weights
        
    print(f"Epoch {epoch+1} complete!")

# 5. SAVE: Save the brain to a file
torch.save(model.state_dict(), 'dental_detective_model.pth')
print("Case Closed. Model Saved.")