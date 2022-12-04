import argparse

import cv2 as cv
import torch
from torchvision.transforms import transforms

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
