import math
import os
import cv2
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.dataset_loader import CSVDataset
from metrics import predict, order_like_split, evaluate
from models import create_model

# Set up experiment
ex = Experiment("train")
ex.observers.append(
    FileStorageObserver.create("results/experiments")
)  # Sacred output folder

# Training settings from each model's original research (see Section 2.2.1 in the report).
# 'lr_decay' is used with an SGD optimiser: the learning rate is multiplied by this value
# when the validation loss has stopped improving.
TRAINING_SETTINGS = {
    "capsule": {"optimizer": "adam", "lr": 0.0005, "betas": (0.9, 0.999)},
    "dsp-fwa": {
        "optimizer": "sgd",
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "lr_decay": 0.9,
    },
    # The original learning rate 0.01 produced NaN loss values, reduced to 0.0001 (Section 3.3)
    "ictu_oculi": {
        "optimizer": "sgd",
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "lr_decay": 0.9,
    },
    "xceptionnet": {"optimizer": "adam", "lr": 0.00001, "betas": (0.5, 0.999)},
}


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
    models_pretrained_path = os.path.join(
        home, "models/pre_trained/"
    )  # path to load pre-trained models
    models_output_path = os.path.join(
        home, "models/re_trained/"
    )  # path to where to save the re-trained models
    train_csv = "train.csv"
    val_csv = "val.csv"
    epochs = 100
    batch_size = 32
    early_stopping = 0
    model_name = None
    split_id = 1


def create_optimizer(model, model_name):
    settings = TRAINING_SETTINGS[model_name]
    parameters = model.trainable_parameters()
    scheduler = None
    if settings["optimizer"] == "adam":
        optimizer = optim.Adam(parameters, lr=settings["lr"], betas=settings["betas"])
    else:
        optimizer = optim.SGD(
            parameters,
            lr=settings["lr"],
            momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=settings["lr_decay"]
        )
    return optimizer, scheduler


# Main function
@ex.automain
def main(
    data_path,
    splits_path,
    output_path,
    models_pretrained_path,
    models_output_path,
    train_csv,
    val_csv,
    epochs,
    batch_size,
    early_stopping,
    model_name,
    split_id,
    _run,
):

    SCORES_DIR = os.path.join(output_path, "model_metrics/train")
    PREDICTIONS_DIR = os.path.join(output_path, "model_predictions/train")
    BEST_MODEL_PATH = os.path.join(models_output_path, model_name + ".pth")

    for folder in (SCORES_DIR, PREDICTIONS_DIR, models_output_path):
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Disable threading to run functions sequentially
    cv2.setNumThreads(0)

    # CPU or GPU utilisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model class and load the pre-trained version from the original authors (transfer learning)
    model = create_model(model_name)
    model.load_pretrained(models_pretrained_path)
    model.to(device)
    optimizer, scheduler = create_optimizer(model, model_name)
    print("Model: {}".format(model_name.upper()))

    # Image transformations: resize the face images to the model's input size
    size = model.input_size[1]
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(model.mean, model.std),
        ]
    )

    # Load datasets
    train_split = pd.read_csv(os.path.join(splits_path, train_csv))
    val_split = pd.read_csv(os.path.join(splits_path, val_csv))
    dataset_train = CSVDataset(
        data_path,
        os.path.join(splits_path, train_csv),
        "frame_id",
        "deepfake",
        transform=transform,
    )
    dataset_val = CSVDataset(
        data_path,
        os.path.join(splits_path, val_csv),
        "frame_id",
        "deepfake",
        transform=transform,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Train model
    best_val_loss = float("inf")
    best = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        print(
            "\nTraining epoch {}/{} for model {}\n".format(
                epoch, epochs, model_name.upper()
            )
        )

        # Train model on training set
        train_loss, train_predictions = predict(
            model, dataloader_train, device, optimizer
        )
        if not math.isfinite(train_loss):
            # See Section 3.3 in the report: the solution was to reduce the learning rate
            raise RuntimeError(
                "The training loss became {} for model {}, try a lower learning rate "
                "(TRAINING_SETTINGS in train.py).".format(train_loss, model_name)
            )
        train_results = evaluate(train_predictions, train_split)

        # Validate model on validation set
        val_loss, val_predictions = predict(model, dataloader_val, device)
        val_results = evaluate(val_predictions, val_split)

        # Accuracy per video (averaged frame scores), loss and AUC per video frame
        epoch_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_results["video_acc"],
            "train_frame_acc": train_results["frame_acc"],
            "train_auc": train_results["frame_auc"],
            "val_loss": val_loss,
            "val_acc": val_results["video_acc"],
            "val_frame_acc": val_results["frame_acc"],
            "val_auc": val_results["frame_auc"],
        }
        history.append(epoch_results)
        for key, value in epoch_results.items():
            if key != "epoch":
                _run.log_scalar(key, value, epoch)
        print(
            "\nTraining loss: {:.4f}, accuracy: {:.2f}%".format(
                train_loss, train_results["video_acc"] * 100
            )
        )
        print(
            "Validation loss: {:.4f}, accuracy: {:.2f}%".format(
                val_loss, val_results["video_acc"] * 100
            )
        )
        print("-" * 40 + "\n")

        if scheduler is not None:
            scheduler.step(val_loss)  # SGD optimiser only

        # Whenever a new best validation loss is achieved, save the model and its predictions
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best = epoch_results
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            order_like_split(train_predictions, train_split).to_csv(
                os.path.join(
                    PREDICTIONS_DIR, "train_predictions_" + model_name + ".csv"
                ),
                index=False,
            )
            order_like_split(val_predictions, val_split).to_csv(
                os.path.join(PREDICTIONS_DIR, "val_predictions_" + model_name + ".csv"),
                index=False,
            )
        else:
            epochs_without_improvement += 1

        # Early stopping if the validation loss hasn't improved for x epochs
        if early_stopping > 0 and epochs_without_improvement >= early_stopping:
            print(
                "No improvement of validation loss for {} epochs, stopping early.".format(
                    early_stopping
                )
            )
            break

    # Save training and validation metrics for every epoch and for the saved (best) model
    print(
        "\nSaving training and validation metrics for model {}\n".format(
            model_name.upper()
        )
    )
    pd.DataFrame(history).to_csv(
        os.path.join(SCORES_DIR, "train_history_" + model_name + ".csv"), index=False
    )
    scores = dict(best)
    scores["best_epoch"] = scores.pop("epoch")
    scores.update(
        {
            "run_id": _run._id,
            "split_id": split_id,
            "model": model_name,
            "epochs_run": len(history),
            "file_size": os.path.getsize(BEST_MODEL_PATH),
        }
    )
    pd.DataFrame([scores]).to_csv(
        os.path.join(SCORES_DIR, "train_scores_" + model_name + ".csv"), index=False
    )

    for artifact in ("train_history_", "train_scores_"):
        ex.add_artifact(os.path.join(SCORES_DIR, artifact + model_name + ".csv"))
    return scores
