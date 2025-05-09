import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    classification_report


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_model_predictions(model, dataloader, device):
    true_labels, predicted_labels = [], []

    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    return true_labels, predicted_labels


def plot_confusion_matrix(model, dataloader, device, class_names):
    true_labels, predicted_labels = _get_model_predictions(model, dataloader,
                                                           device)

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()


def get_classification_report(model, dataloader, device, class_names):
    true_labels, predicted_labels = _get_model_predictions(model, dataloader,
                                                           device)
    print(classification_report(true_labels, predicted_labels,
                                target_names=class_names))


def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from {path}")
    return model
