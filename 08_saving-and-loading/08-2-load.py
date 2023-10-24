import sys
import os

from lightning import Fabric
import torch
import torchmetrics
from torchvision import transforms
from torchvision.models import vit_b_16

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from local_utilities import get_dataloaders_cifar10


test_transforms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()])

_, _, test_loader = get_dataloaders_cifar10(
    batch_size=16,
    num_workers=4,
    train_transforms=None,
    test_transforms=test_transforms,
    validation_fraction=0.1,
    download=True
)

model = vit_b_16(weights=None)
model.heads.head = torch.nn.Linear(in_features=768, out_features=10)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

fabric = Fabric(accelerator="cuda", precision="bf16-mixed", devices=1)
fabric.launch()
test_loader = fabric.setup_dataloaders(test_loader)
model, optimizer = fabric.setup(model, optimizer)

state = {
    "model": model,
    "optimizer": optimizer,
    "anything-else-you-want-to-save": None
}

fabric.load("checkpoint.ckpt", state)

additional_info = state["anything-else-you-want-to-save"]
print("anything-else-you-want-to-save:", additional_info)

with torch.no_grad():
    model.eval()
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    for (features, targets) in test_loader:
        outputs = model(features)
        predicted_labels = torch.argmax(outputs, 1)
        test_acc.update(predicted_labels, targets)

fabric.print(f"Test accuracy {test_acc.compute()*100:.2f}%")
