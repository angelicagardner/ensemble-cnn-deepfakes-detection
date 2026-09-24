import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

from data.average import Average

# A video frame (or video) is classified as deepfake if its score is greater than this value,
# and as real if the score is lesser than or equal to it
THRESHOLD = 0.5


def predict(model, dataloader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    losses = Average()
    frame_ids, labels, scores = [], [], []

    tqdm_loader = tqdm(dataloader)
    for (inputs, targets), names in tqdm_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = model.loss(outputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        frame_ids += list(names)
        labels += targets.cpu().numpy().tolist()
        scores += model.fake_probability(outputs).detach().cpu().numpy().tolist()
        tqdm_loader.set_postfix(loss=losses.avg)

    predictions = pd.DataFrame(
        {"frame_id": frame_ids, "label": labels, "score": scores}
    )
    return losses.avg, predictions


def order_like_split(predictions, split):
    ordered = split[["frame_id"]].merge(predictions, on="frame_id", how="left")
    assert not ordered["score"].isnull().any(), (
        "Predictions are missing for some video frames"
    )
    return ordered


def frame_metrics(predictions):
    labels = predictions["label"].values.astype(int)
    scores = predictions["score"].values
    return {
        "frame_acc": accuracy_score(labels, (scores > THRESHOLD).astype(int)),
        "frame_auc": roc_auc_score(labels, scores),
    }


def video_predictions(predictions, split):
    frames = predictions.merge(
        split[["frame_id", "original_video"]], on="frame_id", how="left"
    )
    videos = frames.groupby("original_video", sort=False).agg(
        label=("label", "first"), score=("score", "mean")
    )
    videos["prediction"] = (videos["score"] > THRESHOLD).astype(int)
    return videos.reset_index()


def evaluate(predictions, split):
    results = frame_metrics(predictions)
    videos = video_predictions(predictions, split)
    tn, fp, fn, tp = confusion_matrix(
        videos["label"], videos["prediction"], labels=[0, 1]
    ).ravel()
    results.update(
        {
            "videos": len(videos),
            "video_acc": accuracy_score(videos["label"], videos["prediction"]),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        }
    )
    return results


def roc_points(predictions):
    fpr, tpr, thresholds = roc_curve(
        predictions["label"].values.astype(int), predictions["score"].values
    )
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
