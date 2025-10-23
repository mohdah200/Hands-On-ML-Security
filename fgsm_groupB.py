
#!/usr/bin/env python
# coding: utf-8

import os, json, random, csv
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

SEED = int(os.getenv("SEED", "42"))
BATCH = int(os.getenv("BATCH", "128"))
TEST_SAMPLES = int(os.getenv("TEST_SAMPLES", "5000"))
EPS = float(os.getenv("EPS", "0.2"))
SURR_TRAIN_SAMPLES = int(os.getenv("SURR_TRAIN_SAMPLES", "10000"))
SURR_EPOCHS = int(os.getenv("SURR_EPOCHS", "1"))
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
train_subset = Subset(train_ds, list(range(SURR_TRAIN_SAMPLES)))

train_loader = DataLoader(train_subset, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_subset,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# 2) Victim architecture must match Group A
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

victim = SimpleCNN().to(DEVICE)

# 3) Load Group A weights and metadata
art_dir = os.getenv("ARTIFACTS_A_DIR", "artifacts_A")
weights_path = os.path.join(art_dir, "model_A.pth")
meta_path = os.path.join(art_dir, "meta_A.json")

assert os.path.exists(weights_path), f"Missing {weights_path}"
victim.load_state_dict(torch.load(weights_path, map_location=DEVICE))

meta = {}
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

# 4) Helper for accuracy
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

# 5) Verify clean accuracy of the victim on the same subset
clean_acc = accuracy(victim, test_loader)
print(f"Victim clean accuracy on {TEST_SAMPLES}: {clean_acc:.4f}")

# 6) Build a different surrogate model
# Use a simple MLP to increase architectural diversity vs SimpleCNN
class SurrogateMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

surr = SurrogateMLP().to(DEVICE)
opt_surr = optim.Adam(surr.parameters(), lr=1e-3)
crit_surr = nn.CrossEntropyLoss()

# 7) Train surrogate briefly
surr.train()
for epoch in range(SURR_EPOCHS):
    run_loss = 0.0
    for i, (x, y) in enumerate(train_loader, start=1):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt_surr.zero_grad()
        out = surr(x)
        loss = crit_surr(out, y)
        loss.backward()
        opt_surr.step()
        run_loss += loss.item()
        if i % 200 == 0:
            print(f"surr epoch {epoch+1} step {i} loss {run_loss/i:.4f}")

# 8) Wrap surrogate with ART and craft FGSM
surr_clf = PyTorchClassifier(
    model=surr,
    clip_values=(0.0, 1.0),
    loss=crit_surr,
    optimizer=opt_surr,
    input_shape=(1, 28, 28),
    nb_classes=10
)

def fgsm_generate_on_loader(clf, loader, eps, batch=BATCH):
    atk = FastGradientMethod(estimator=clf, eps=eps, batch_size=batch)
    adv_imgs, adv_lbls = [], []
    for x, y in loader:
        x_np = x.cpu().numpy()
        x_adv_np = atk.generate(x=x_np)
        adv_imgs.append(torch.tensor(x_adv_np, dtype=x.dtype))
        adv_lbls.append(y.clone())
    adv_x = torch.cat(adv_imgs, dim=0)
    adv_y = torch.cat(adv_lbls, dim=0)
    return adv_x, adv_y

adv_x, adv_y = fgsm_generate_on_loader(surr_clf, test_loader, EPS)

# 9) Evaluate victim on transfer adversarial set
adv_loader_for_victim = DataLoader(TensorDataset(adv_x, adv_y), batch_size=BATCH, shuffle=False)
transfer_acc = accuracy(victim, adv_loader_for_victim)
attack_success = 1.0 - transfer_acc

print(f"Transfer FGSM eps {EPS:.3f}: victim accuracy {transfer_acc:.4f}")
print(f"Attack success rate {attack_success:.4f}")

if attack_success >= 0.25:
    print("Nice transfer. Team B fooled Team A on many samples. 🎉")
else:
    print("Victim resisted a good portion of transfers. Team A holds up. 💪")

# 10) Save a summary CSV
os.makedirs("artifacts_B", exist_ok=True)
with open(os.path.join("artifacts_B", "results_B.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["victim_clean_accuracy", round(clean_acc, 4)])
    w.writerow(["fgsm_eps", EPS])
    w.writerow(["transfer_victim_accuracy", round(transfer_acc, 4)])
    w.writerow(["attack_success_rate", round(attack_success, 4)])

print("Saved artifacts_B/results_B.csv")
print("Done.")
