import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


def open_resize_save_image(input_path, output_path, new_size):
    try:
        # Open the image file
        image = Image.open(input_path)

        # Resize the image to the desired size
        resized_image = image.resize(new_size)

        # Save the resized image to the output path
        resized_image.save(output_path)

        # print("Image successfully resized and saved.")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_datasets(concepts, config):
    if not os.path.isdir(config["CAV_data"]["concept_output_path"]):
        os.mkdir(config["CAV_data"]["concept_output_path"])

    # Initialize the 'test' list in config["data"]
    config["data"] = {}
    config["data"]["test"] = []
    # Generic dataloader dictionary to add to config
    generic_dataloader = {
        "type": "DataLoader",
        "batch_size": 4096,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 4,
    }
    # Loop through selected identities and create datasets
    for concept in tqdm(concepts):
        current_concept_dataset = {
            "type": "ConceptDataset",
            "name": "_".join(concept.split(" ")),
            "data_dir": config["CAV_data"]["concept_output_path"],
            "ann_path": f"{config['CAV_data']['concept_output_path']}{'_'.join(concept.split(' '))}_ann.txt",
            "test_mode": True,
        }
        config["data"]["test"].append(
            {"dataset": current_concept_dataset, "dataloader": generic_dataloader}
        )

        # Create identity directory if it doesn't exist
        data_dir_path = current_concept_dataset["data_dir"] + "_".join(
            concept.split(" ")
        )
        if not os.path.isdir(data_dir_path):
            os.mkdir(data_dir_path)

            id_images = os.listdir(config["CAV_data"]["concept_data_path"] + concept)
            id_images = [f"{concept}/{img}" for img in id_images]

            # Process and save images
            ann_output_list = []
            for image in id_images:
                input_path = config["CAV_data"]["concept_data_path"] + image
                output_path = (
                    current_concept_dataset["data_dir"]
                    + "_".join(image.split(" "))[:-3]
                    + "bmp"
                )
                open_resize_save_image(input_path, output_path, (112, 112))
                ann_output_list.append("/".join(output_path.split("/")[-2:]))
            ann_output_df = pd.DataFrame(ann_output_list)
            ann_output_df.to_csv(
                current_concept_dataset["ann_path"], sep=" ", index=None, header=None
            )
    return config


def create_CAVs(model_embeddings, config):
    for concept in tqdm(model_embeddings):
        if os.path.isfile(f"CAVs/{config['CAV_data']['source_name']}/{concept}.pth"):
            continue

        concept_samples = model_embeddings[concept].to("cuda")
        concept_labels = torch.ones(concept_samples.shape[0], dtype=torch.float).to(
            "cuda"
        )

        random_samples = []
        for i in model_embeddings:
            if i != concept:
                random_samples.append(model_embeddings[i])
        random_samples = torch.cat(random_samples).to("cuda")
        random_labels = torch.zeros(random_samples.shape[0], dtype=torch.float).to(
            "cuda"
        )

        classifier_inputs = torch.cat([concept_samples, random_samples])
        classifier_labels = torch.cat([concept_labels, random_labels])

        binary_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(
            "cuda"
        )
        binary_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5).to("cuda")

        # classifier = nn.Linear(in_features=concept_samples.shape[1], out_features=1).to("cuda")
        classifier = LinearClassifier(concept_samples.shape[1], 1).to("cuda")

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(classifier.parameters(), lr=1000.0, momentum=0.9)
        # optimizer = optim.SGD(classifier.parameters(), lr=10., momentum=0.9)

        # Training loop
        acc_flag = True
        iteration = 0
        while acc_flag and (iteration < 1000000):
            iteration += 1
            optimizer.zero_grad()
            logits = torch.squeeze(classifier(classifier_inputs))
            loss = loss_fn(logits, classifier_labels)
            loss.backward()
            optimizer.step()

            predictions = torch.logical_not(torch.lt(logits, 0.5)).float()
            acc = binary_acc(predictions, classifier_labels)
            f1 = binary_f1(predictions, classifier_labels)
            if acc >= 1:
                break
            # if iteration % 1000 == 0:
            #     print(iteration, acc, f1, loss.item())
        # print(f"Iteration: {iteration}, Acc.: {acc}, F1: {f1}")
        torch.save(
            classifier.state_dict(),
            f"CAVs/{config['CAV_data']['source_name']}/{concept}.pth",
        )


def load_CAVs(selected_concepts, config):
    cav_dict = {}
    for concept in selected_concepts:
        CAV = LinearClassifier(512, 1)
        CAV.load_state_dict(
            torch.load(f"CAVs/{config['CAV_data']['source_name']}/{concept}.pth")
        )
        CAV = CAV.to("cuda")
        CAV.eval()
        cav_dict[concept] = CAV
    return cav_dict


def calculate_concept_centers(embeddings_df):
    unique_class_names = embeddings_df["class"].unique()
    concept_centers = np.zeros(
        (
            len(unique_class_names),
            embeddings_df.drop(["class", "path", "class_num"], axis=1).shape[1],
        )
    )
    for class_name_index, class_name in enumerate(unique_class_names):
        class_average = (
            embeddings_df[embeddings_df["class"] == class_name]
            .drop(["class", "path", "class_num"], axis=1)
            .mean(axis=0)
            .values
        )
        concept_centers[class_name_index] = class_average

    concept_centers_df = pd.DataFrame(concept_centers)
    concept_centers_df["class"] = unique_class_names

    return concept_centers_df


def calculate_concept_center_distances(concept_centers_df, distance_metric="euclidean"):
    # Calculate pairwise distances using pdist
    distances = pdist(
        concept_centers_df.drop(["class"], axis=1).values, metric=distance_metric
    )

    # Convert the condensed distance matrix to a square matrix
    distance_matrix = squareform(distances)

    # Create a new DataFrame with distances
    concept_distance_df = pd.DataFrame(
        distance_matrix,
        index=concept_centers_df["class"],
        columns=concept_centers_df["class"],
    )

    return concept_distance_df


def concept_center_analysis(concept_distance_df, doppelganger_pairs):
    analysis_list = [
        [
            "Name",
            "Doppelganger_Name",
            "Dist_doppelganger",
            "Min_Name",
            "Dist_Min",
            "Max_Name",
            "Dist_Max",
        ]
    ]
    for index, row in doppelganger_pairs.iterrows():
        concept1 = row["Pair 1"]
        concept2 = row["Pair 2"]
        dist_to_doppelganger = concept_distance_df[concept1][concept2]

        concept1_max_name = concept_distance_df[concept1].idxmax()
        concept1_max = concept_distance_df[concept1][concept1_max_name]
        concept1_min_name = concept_distance_df[concept1].drop([concept1]).idxmin()
        concept1_min = concept_distance_df[concept1][concept1_min_name]

        analysis_list.append(
            [
                concept1,
                concept2,
                dist_to_doppelganger,
                concept1_min_name,
                concept1_min,
                concept1_max_name,
                concept1_max,
            ]
        )

        concept2_max_name = concept_distance_df[concept2].idxmax()
        concept2_max = concept_distance_df[concept2][concept2_max_name]
        concept2_min_name = concept_distance_df[concept2].drop([concept2]).idxmin()
        concept2_min = concept_distance_df[concept2][concept2_min_name]

        analysis_list.append(
            [
                concept2,
                concept1,
                dist_to_doppelganger,
                concept2_min_name,
                concept2_min,
                concept2_max_name,
                concept2_max,
            ]
        )

    concept_center_analysis_df = pd.DataFrame(analysis_list)
    return concept_center_analysis_df


def compute_top_n_per_doppelganger_pair(
    cluster_distance_df, doppelganger_pairs_df, top_n_to_check
):
    assert isinstance(top_n_to_check, list), "'top_n_to_check' must be a list."
    top_n_analysis_list = []
    for index, row in doppelganger_pairs_df.iterrows():
        query_row_sorted = cluster_distance_df[row["Pair 1"]].sort_values()

        row_list = [row["Pair 1"], row["Pair 2"], query_row_sorted.index[1]]
        for n in top_n_to_check:
            query_row_sorted_reduced = query_row_sorted[: n + 1]
            prediction = row["Pair 2"] in query_row_sorted_reduced
            row_list += [prediction]
        top_n_analysis_list.append(row_list)
    doppelganger_pair_top_n_analysis_df = pd.DataFrame(
        top_n_analysis_list,
        columns=["Source ID", "Doppelganger ID", "Closest ID"]
        + [f"Top {n} Acc." for n in top_n_to_check],
    )

    acc_list = []
    for column in [f"Top {n} Acc." for n in top_n_to_check]:
        acc_list.append(
            doppelganger_pair_top_n_analysis_df[column].sum()
            / doppelganger_pair_top_n_analysis_df.shape[0]
        )

    return doppelganger_pair_top_n_analysis_df, acc_list


def compute_top_n_per_sample(
    concept_centers_df, embeddings_df, top_n_to_check, distance_metric="euclidean"
):
    assert isinstance(top_n_to_check, list), "'top_n_to_check' must be a list."

    # Extract the N-dimensional arrays from DataFrames
    embeddings_array = embeddings_df.drop(["class", "path", "class_num"], axis=1).values
    concept_center_array = concept_centers_df.drop(["class"], axis=1).values

    # Compute the distance matrix using cdist
    distances = cdist(
        embeddings_array, concept_center_array, distance_metric
    )  # You can replace 'euclidean' with other distance metrics

    # Create a new DataFrame with the distances
    point_to_center_distances_df = pd.DataFrame(
        distances, columns=concept_centers_df["class"]
    )

    # Concatenate the distances DataFrame with the original points DataFrame
    # result_df = pd.concat([embeddings_df, distances_df], axis=1)

    top_n_analysis_list = []
    for index, row in embeddings_df.iterrows():
        query_row_sorted = point_to_center_distances_df.loc[index].sort_values()

        row_list = [row["class"], query_row_sorted.index[0]]
        for n in top_n_to_check:
            query_row_sorted_reduced = query_row_sorted[:n]
            prediction = row["class"] in query_row_sorted_reduced
            row_list += [prediction]
        top_n_analysis_list.append(row_list)
    sample_top_n_analysis_df = pd.DataFrame(
        top_n_analysis_list,
        columns=["Source ID", "Closest ID"] + [f"Top {n} Acc." for n in top_n_to_check],
    )

    acc_list = []
    for column in [f"Top {n} Acc." for n in top_n_to_check]:
        acc_list.append(
            sample_top_n_analysis_df[column].sum() / sample_top_n_analysis_df.shape[0]
        )

    per_id_sample_top_n_analysis_list = []
    for unique_id in sample_top_n_analysis_df["Source ID"].unique():
        id_df = sample_top_n_analysis_df[
            sample_top_n_analysis_df["Source ID"] == unique_id
        ]

        current_sample_list = [unique_id]
        for column in [f"Top {n} Acc." for n in top_n_to_check]:
            current_sample_list.append(id_df[column].sum() / id_df.shape[0])

        per_id_sample_top_n_analysis_list.append(current_sample_list)

    per_id_sample_top_n_analysis_df = pd.DataFrame(
        per_id_sample_top_n_analysis_list,
        columns=["ID"] + [f"Top {n} Acc." for n in top_n_to_check],
    )

    # temp = per_id_sample_top_n_analysis_df.describe()

    return sample_top_n_analysis_df, acc_list
