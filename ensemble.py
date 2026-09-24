import os
import numpy as np
import pandas as pd
import torch
from deepstack.base import Member
from deepstack.ensemble import StackEnsemble, DirichletEnsemble

from sacred import Experiment
from sacred.observers import FileStorageObserver

# DeepStack loads TensorFlow (through Keras) and Sacred 0.7.4 seeds TensorFlow with the TensorFlow 1
# function set_random_seed, which is called tf.random.set_seed in TensorFlow 2
import tensorflow as tf

if not hasattr(tf, "set_random_seed"):
    tf.set_random_seed = tf.random.set_seed

from metrics import THRESHOLD, evaluate, roc_points
from models import MODELS, create_model

# Set up experiment
ex = Experiment("ensemble")
ex.observers.append(
    FileStorageObserver.create("results/experiments")
)  # Sacred output folder

# The six ensembles in the experiment (Table 3.1 in the report):
# which base-learners are combined, and hard (majority) or soft (weighted) voting
ENSEMBLES = [
    ("best_hard", "best", "hard"),
    ("best_soft", "best", "soft"),
    ("small_hard", "small", "hard"),
    ("small_soft", "small", "soft"),
    ("all_hard", "all", "hard"),
    ("all_soft", "all", "soft"),
]


# Add default configurations
@ex.config
def cfg():
    home = os.getcwd()

    splits_path = os.path.join(
        home, "data/splits/"
    )  # path to CSV files with information about train, validation, and test splits
    output_path = os.path.join(
        home, "results/"
    )  # path to output folder where the results should be stored
    models_saved_path = os.path.join(
        home, "models/re_trained/"
    )  # path to the saved re-trained single models
    models_train_predictions_path = os.path.join(
        output_path, "model_predictions/train/"
    )  # single model predictions on the training and validation sets
    models_test_predictions_path = os.path.join(
        output_path, "model_predictions/test/"
    )  # single model predictions on the test set
    models_train_metrics_path = os.path.join(
        output_path, "model_metrics/train/"
    )  # single model training and validation metrics
    test_csv = "test.csv"  # test CSV file
    seed = 1  # random seed (also used by Sacred), e.g. for the weight search in DirichletEnsemble


class MajorityVote(object):
    def fit(self, X, y, **kwargs):
        return self  # nothing to learn

    def predict(self, X, **kwargs):
        votes = (np.asarray(X, dtype=float) > THRESHOLD).astype(float)
        return votes.mean(axis=1)


def load_member(
    model_name,
    models_saved_path,
    train_predictions_path,
    test_predictions_path,
    train_metrics_path,
):
    model_path = os.path.join(models_saved_path, model_name + ".pth")

    # Load the saved single model and change it to evaluation mode before adding it to the ensemble
    model = create_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    train = pd.read_csv(
        os.path.join(train_predictions_path, "train_predictions_" + model_name + ".csv")
    )
    val = pd.read_csv(
        os.path.join(train_predictions_path, "val_predictions_" + model_name + ".csv")
    )
    test = pd.read_csv(
        os.path.join(test_predictions_path, "test_predictions_" + model_name + ".csv")
    )
    train_scores = pd.read_csv(
        os.path.join(train_metrics_path, "train_scores_" + model_name + ".csv")
    )

    member = Member(
        name=model_name.capitalize(),
        train_probs=train["score"].values,
        train_classes=train["label"].values,
        val_probs=val["score"].values,
        val_classes=val["label"].values,
        submission_probs=test["score"].values,
    )
    return {
        "name": model_name,
        "member": member,
        "model": model,
        "test": test,
        "val_frame_ids": val["frame_id"].values,
        "val_acc": float(train_scores["val_acc"].iloc[0]),
        "file_size": os.path.getsize(model_path),
    }


def select_members(single_models, selection):
    if selection == "best":
        return sorted(single_models, key=lambda m: m["val_acc"], reverse=True)[:2]
    if selection == "small":
        return sorted(single_models, key=lambda m: m["file_size"])[:2]
    return list(single_models)


# Main function
@ex.automain
def main(
    splits_path,
    output_path,
    models_saved_path,
    models_train_predictions_path,
    models_test_predictions_path,
    models_train_metrics_path,
    test_csv,
    seed,
    _run,
):

    ENSEMBLE_DIR = os.path.join(output_path, "ensemble")
    if not os.path.exists(ENSEMBLE_DIR):
        os.makedirs(ENSEMBLE_DIR)

    test_split = pd.read_csv(os.path.join(splits_path, test_csv))

    # Load the base-learners (all saved single models)
    single_models = [
        load_member(
            name,
            models_saved_path,
            models_train_predictions_path,
            models_test_predictions_path,
            models_train_metrics_path,
        )
        for name in MODELS
        if os.path.exists(os.path.join(models_saved_path, name + ".pth"))
    ]
    print(
        "Loaded base-learners: {}".format(", ".join(m["name"] for m in single_models))
    )
    for m in single_models[1:]:
        assert (
            m["test"]["frame_id"].values == single_models[0]["test"]["frame_id"].values
        ).all()
        assert (m["val_frame_ids"] == single_models[0]["val_frame_ids"]).all()
    test_frames = single_models[0]["test"][["frame_id", "label"]]

    # Single model test performances (for comparison with the ensembles)
    single_results = {m["name"]: evaluate(m["test"], test_split) for m in single_models}

    summary = []
    for ensemble_name, selection, voting in ENSEMBLES:
        print("\n" + "-" * 40 + "\nEnsemble: {}\n".format(ensemble_name))
        members = select_members(single_models, selection)

        # Initialize ensemble (hard/majority vs. soft/weighted voting) and add members
        if voting == "hard":
            ensemble = StackEnsemble(model=MajorityVote())
        else:
            np.random.seed(seed)
            ensemble = DirichletEnsemble()
        ensemble.add_members([m["member"] for m in members])

        # Train the ensemble (DirichletEnsemble: weights are optimised on the validation set)
        ensemble.fit()
        print("Validation set (DeepStack):")
        ensemble.describe()

        # Ensemble predictions on the test set, evaluated the same way as the single models
        predictions = test_frames.copy()
        predictions["score"] = ensemble.predict()
        results = evaluate(predictions, test_split)

        rows = []
        for i, m in enumerate(members):
            row = {
                "base_learner": m["name"],
                "file_size": m["file_size"],
                "val_acc": m["val_acc"],
            }
            row["weight"] = ensemble.bestweights[i] if voting == "soft" else np.nan
            row.update(single_results[m["name"]])
            rows.append(row)
        rows.append(dict(base_learner="ensemble", weight=np.nan, **results))
        pd.DataFrame(rows).to_csv(
            os.path.join(ENSEMBLE_DIR, ensemble_name + ".csv"), index=False
        )
        roc_points(predictions).to_csv(
            os.path.join(ENSEMBLE_DIR, ensemble_name + "_roc.csv"), index=False
        )
        predictions.to_csv(
            os.path.join(ENSEMBLE_DIR, ensemble_name + "_predictions.csv"), index=False
        )
        ex.add_artifact(os.path.join(ENSEMBLE_DIR, ensemble_name + ".csv"))

        print("\nTest set:")
        print(
            pd.DataFrame(rows)[
                [
                    "base_learner",
                    "weight",
                    "frame_auc",
                    "video_acc",
                    "sensitivity",
                    "specificity",
                ]
            ]
        )
        for key in ("frame_auc", "video_acc", "sensitivity", "specificity"):
            _run.log_scalar(ensemble_name + "." + key, results[key])
        summary.append(
            dict(
                ensemble=ensemble_name,
                members=", ".join(m["name"] for m in members),
                voting=voting,
                **results,
            )
        )

    pd.DataFrame(summary).to_csv(os.path.join(ENSEMBLE_DIR, "summary.csv"), index=False)
    ex.add_artifact(os.path.join(ENSEMBLE_DIR, "summary.csv"))
