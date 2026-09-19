import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch

from rbj.models.vision import SJGVision
from rbj.perception.vision.vision import VisionDatasetGenerator


def image_to_tensor(image):

    tensor = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    )

    tensor = tensor.reshape(
        image.height,
        image.width,
        3
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor / 255.0

    return tensor.unsqueeze(0)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SJGVision().to(device)

    model.load_state_dict(
        torch.load(
            "checkpoints/sjg_vision_02.pt",
            map_location=device
        )
    )

    model.eval()

    generator = VisionDatasetGenerator(
        image_size=64,
        grid_size=5
    )

    image, real_position = (
        generator.generate()
    )

    x = image_to_tensor(
        image
    ).to(device)

    with torch.no_grad():
        output = model(x)
        prediction = output[0].tolist()

    norm = max(1, generator.grid_size - 1)
    predicted_coords = tuple(int(round(p * norm)) for p in prediction)
    real_coords = tuple(int(round(p * norm)) for p in real_position)

    print("Estado real (Agente, Objetivo, Obstáculo):")
    print(
        f"Agente: ({real_coords[0]}, {real_coords[1]}) | "
        f"Objetivo: ({real_coords[2]}, {real_coords[3]}) | "
        f"Obstáculo: ({real_coords[4]}, {real_coords[5]})"
    )

    print("\nEstado predicho:")
    print(
        f"Agente: ({predicted_coords[0]}, {predicted_coords[1]}) | "
        f"Objetivo: ({predicted_coords[2]}, {predicted_coords[3]}) | "
        f"Obstáculo: ({predicted_coords[4]}, {predicted_coords[5]})"
    )

    image.resize(
        (512, 512)
    ).save(
        "vision_test.png"
    )


if __name__ == "__main__":
    main()