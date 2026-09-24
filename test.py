import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.dataset_loader import CSVDataset
from metrics import predict, order_like_split, evaluate, roc_points
from models import create_model

# Set up experiment
ex = Experiment("test")
ex.observers.append(
    FileStorageObserver.create("results/experiments")
)  # Sacred output folder


# Add default configurations
@ex.config
def cfg():
    home = os.getcwd()

    data_path = os.path.join(
        home, "data/images/"
    )  # path to video frames (folder containing images)
    splits_path = os.path.join(
        home, "data/splits/"
    )  # path to CSV files with information about train, validation, and test splits
    output_path = os.path.join(
        home, "results/"
    )  # path to output folder where the results should be stored
    models_retrained_path = os.path.join(
        home, "models/re_trained/"
    )  # path to load the re-trained models
    test_csv = "test.csv"
    batch_size = 32
    model_name = None


def test(model, data_path, splits_path, test_csv, batch_size, device):
    size = model.input_size[1]
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(model.mean, model.std),
        ]
    )

    split = pd.read_csv(os.path.join(splits_path, test_csv))
    dataset = CSVDataset(
        data_path,
        os.path.join(splits_path, test_csv),
        "frame_id",
        "deepfake",
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluation mode (PyTorch models are in training mode by default)
    model.eval()
    model.to(device)
    loss, predictions = predict(model, dataloader, device)
    predictions = order_like_split(predictions, split)

    results = evaluate(predictions, split)
    results["loss"] = loss
    return results, predictions


# Main function
@ex.automain
def main(
    data_path,
    splits_path,
    output_path,
    models_retrained_path,
    test_csv,
    batch_size,
    model_name,
    _run,
):

    METRICS_DIR = os.path.join(output_path, "model_metrics/test")
    PREDICTIONS_DIR = os.path.join(output_path, "model_predictions/test")
    for folder in (METRICS_DIR, PREDICTIONS_DIR):
        if not os.path.exists(folder):
            os.makedirs(folder)

    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the re-trained model (best version saved during training)
    model = create_model(model_name)
    model.load_state_dict(
        torch.load(
            os.path.join(models_retrained_path, model_name + ".pth"), map_location="cpu"
        )
    )
    print("Model: {}".format(model_name.upper()))

    # Make predictions on test set
    results, predictions = test(
        model, data_path, splits_path, test_csv, batch_size, device
    )
    results["model"] = model_name
    print(results)

    # Save predictions and evaluation metrics
    predictions.to_csv(
        os.path.join(PREDICTIONS_DIR, "test_predictions_" + model_name + ".csv"),
        index=False,
    )
    pd.DataFrame([results]).to_csv(
        os.path.join(METRICS_DIR, "test_scores_" + model_name + ".csv"), index=False
    )
    roc_points(predictions).to_csv(
        os.path.join(METRICS_DIR, "test_roc_" + model_name + ".csv"), index=False
    )

    for key, value in results.items():
        if key != "model":
            _run.log_scalar(key, value)
    for prefix in ("test_scores_", "test_roc_"):
        ex.add_artifact(os.path.join(METRICS_DIR, prefix + model_name + ".csv"))
    return results
