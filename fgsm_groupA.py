
#!/usr/bin/env python
# coding: utf-8

import os, json, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

SEED = int(os.getenv("SEED", "42"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
BATCH = int(os.getenv("BATCH", "128"))
TEST_SAMPLES = int(os.getenv("TEST_SAMPLES", "5000"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1) Data
train_tf = T.Compose([T.ToTensor()])
test_tf  = T.Compose([T.ToTensor()])
root = "./data"
train_ds = torchvision.datasets.MNIST(root=root, train=True,  download=True, transform=train_tf)
test_ds  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=test_tf)

test_subset = Subset(test_ds, list(range(TEST_SAMPLES)))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_subset, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# 2) Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3) Train
model.train()
for epoch in range(EPOCHS):
    run_loss = 0.0
    for i, (x, y) in enumerate(train_loader, start=1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        if i % 200 == 0:
            print(f"epoch {epoch+1} step {i} loss {run_loss/i:.4f}")

# 4) Clean accuracy on the fixed subset
def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

clean_acc = accuracy(model, test_loader)
print(f"Clean accuracy on {TEST_SAMPLES}: {clean_acc:.4f}")

# 5) Save artifacts for Group B
os.makedirs("artifacts_A", exist_ok=True)
weights_path = os.path.join("artifacts_A", "model_A.pth")
torch.save(model.state_dict(), weights_path)

meta = {
    "seed": SEED,
    "epochs": EPOCHS,
    "batch": BATCH,
    "test_samples": TEST_SAMPLES,
    "clean_accuracy": round(clean_acc, 4),
    "victim_arch": "SimpleCNN",
    "weights_file": "model_A.pth"
}
with open(os.path.join("artifacts_A", "meta_A.json"), "w") as f:
    json.dump(meta, f, indent=2)

# also save deterministic indices for the eval subset in case students want to double check
np.save(os.path.join("artifacts_A", "eval_indices.npy"), np.arange(TEST_SAMPLES, dtype=np.int64))

print("Saved:")
print(f" - {weights_path}")
print(" - artifacts_A/meta_A.json")
print(" - artifacts_A/eval_indices.npy")
print("Share the artifacts_A folder with Group B.")
