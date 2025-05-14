import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Model Definition (Must match training)
# -----------------------------
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=1),
            nn.LeakyReLU(0.1)
        )
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, embed=False):
        x = self.conv_layers(x)
        x = self.feature_extractor(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        if embed:
            return x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel().to(device)
model.load_state_dict(torch.load("model_epoch_20.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 65535.0)
])

def get_embedding(image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img, embed=True).cpu().numpy().flatten()
    return emb

# -----------------------------
# Load images from folder
# -----------------------------
folder = "../all_598_augmented"
images_by_class = defaultdict(list)

for fname in os.listdir(folder):
    if not fname.endswith(".png"):
        continue
    path = os.path.join(folder, fname)
    label = "malignant" if "MALIGNANT" in fname.upper() else "benign"
    images_by_class[label].append(path)

# -----------------------------
# One-Shot Inference (1 per class)
# -----------------------------
print("\n One-Shot Results:")
support_embeddings = []
support_labels = []

# Take 1 support image per class
for label in images_by_class:
    support_path = images_by_class[label][0]
    support_embeddings.append(get_embedding(support_path))
    support_labels.append(label)

# Query: take next 5 images per class
query_images = [img for paths in images_by_class.values() for img in paths[1:6]]

for qpath in query_images:
    qemb = get_embedding(qpath)
    sims = cosine_similarity([qemb], support_embeddings)[0]
    pred = support_labels[np.argmax(sims)]
    print(f"{os.path.basename(qpath)} ➜ {pred}")

# -----------------------------
# Few-Shot Inference (5-shot)
# -----------------------------
print("\n Few-Shot (5-shot) Results:")
K = 5
prototypes = {}

for label in images_by_class:
    embs = [get_embedding(p) for p in images_by_class[label][:K]]
    prototypes[label] = np.mean(embs, axis=0)

for qpath in query_images:
    qemb = get_embedding(qpath)
    sims = {label: cosine_similarity([qemb], [proto])[0][0] for label, proto in prototypes.items()}
    pred = max(sims.items(), key=lambda x: x[1])[0]
    print(f"{os.path.basename(qpath)} ➜ {pred}")


one_shot_preds = []
few_shot_preds = []

for qpath in query_images:
    qname = os.path.basename(qpath)
    qemb = get_embedding(qpath)

    # One-shot prediction
    sims_1 = cosine_similarity([qemb], support_embeddings)[0]
    one_shot_pred = support_labels[np.argmax(sims_1)]

    # Few-shot prediction
    sims_k = {label: cosine_similarity([qemb], [proto])[0][0] for label, proto in prototypes.items()}
    few_shot_pred = max(sims_k.items(), key=lambda x: x[1])[0]

    one_shot_preds.append((qname, one_shot_pred))
    few_shot_preds.append((qname, few_shot_pred))

# -----------------------------
# Create DataFrame for plotting
# -----------------------------
df = pd.DataFrame({
    'Image': [os.path.basename(p) for p in query_images],
    '1-Shot': [p[1] for p in one_shot_preds],
    '5-Shot': [p[1] for p in few_shot_preds]
})

# -----------------------------
# Plot and save as PNG
# -----------------------------
fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)
table.scale(1, 1.5)
plt.title("One-Shot vs Few-Shot Predictions", fontweight='bold')
plt.tight_layout()
plt.savefig("one_few_shot_results.png", dpi=300)
print("✅ Saved: one_few_shot_results.png")