import argparse

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def infer(data, network, loss_fn, device_type):
    x_cpu, y_cpu = data
    x = x_cpu.to(device_type).float()
    y = y_cpu.to(device_type).long()
    output = network(x)
    loss = loss_fn(output, y)
    return output, loss


def evaluate(
        network: nn.Module,
        test_data: DataLoader,
        loss_fn,
        device_type: str
) -> [np.ndarray, np.ndarray, list[float]]:
    """
    Test a given model and return true, predicted values and loss
    """
    network.eval()
    preds, losses = np.array([]), []
    trues = np.array([])
    with torch.no_grad():
        for data in test_data:
            output, loss = infer(data, network, loss_fn, device_type)
            trues = np.concatenate((trues, data[1].data.numpy()))
            preds = np.concatenate(
                (preds, torch.nn.functional.softmax(output, dim=1)
                 .cpu()
                 .data
                 .numpy()
                 .argmax(axis=1))
            )
            losses.append(loss.item())
    return trues, preds, losses


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Load a model and run inference on a source image")
    parser.add_argument(
        "-m", "--model", required=True, type=str, help="path to a pickled model to load"
    )
    parser.add_argument \
        ("-i", "--image", required=True, type=str, help="path to an image to load"
         )
    args = parser.parse_args()
    model = torch.load(args.model)
    image = cv.imread(args.image)
    image = image / 255
    processed = transforms.ToTensor()(image).to(device)
    predicted = model(processed.float().unsqueeze(0))
    labels = {
        '0': 'ArtDeco',
        '1': 'Classic',
        '2': 'Glamour',
        '3': 'Industrial',
        '4': 'Minimalistic',
        '5': 'Modern',
        '6': 'Rustic',
        '7': 'Scandinavian',
        '8': 'Vintage',
    }
    print(labels[str(
        torch.nn.functional.softmax(predicted, dim=1).cpu().data.numpy().argmax(axis=1)[0]
    )])
