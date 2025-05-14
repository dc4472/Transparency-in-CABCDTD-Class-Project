import torch
import matplotlib.pyplot as plt
import os

# Paths
model_dir = './'
epoch_range = range(11, 21)

# Lists to collect accuracy and epoch
epochs = []
accuracies = []

# Iterate through models
for epoch in epoch_range:
    model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
    print(f"🔍 Loading model: {model_path}")
    
    if os.path.exists(model_path):
        model_state = torch.load(model_path, map_location='cpu')

        if 'accuracy' in model_state:
            acc = model_state['accuracy']
            print(f"Epoch {epoch}: Accuracy = {acc:.2f}")
            epochs.append(epoch)
            accuracies.append(acc)
        else:
            print(f"Warning: Accuracy not found in model_epoch_{epoch}.pth")
    else:
        print(f"Model file not found: model_epoch_{epoch}.pth")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='teal')
plt.title("Training Accuracy Across Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("epoch_accuracy_plot.png", dpi=300)
print("📊 Saved: epoch_accuracy_plot.png")
