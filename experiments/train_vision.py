import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rbj.models.vision import SJGVision
from rbj.perception.vision.vision import VisionDatasetGenerator


class VisionDataset(Dataset):
    def __init__(self, generator, samples):
        self.generator = generator
        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        image, position = self.generator.generate()

        tensor = torch.tensor(
            list(image.getdata()),
            dtype=torch.float32
        )

        tensor = tensor.reshape(
            image.height,
            image.width,
            3
        ).permute(2, 0, 1) / 255.0

        label = torch.tensor(list(position), dtype=torch.float32)

        return tensor, label


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    generator = VisionDatasetGenerator(
        image_size=64,
        grid_size=5
    )

    print("Preparando DataLoaders...")

    train_dataset = VisionDataset(generator, 10000)
    test_dataset = VisionDataset(generator, 2000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0
    )

    model = SJGVision().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / len(train_loader):.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        "checkpoints/sjg_vision_02.pt"
    )

    print("Modelo guardado exitosamente.")


if __name__ == "__main__":
    main()
