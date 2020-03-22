import matplotlib.pyplot as plt
import numpy as np
import umap
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


COLOR_DICT = {
    0: "blue",
    1: "orange",
    2: "green",
    3: "red",
    4: "purple",
    5: "brown",
    6: "olive",
    7: "gray",
    8: "pink",
    9: "cyan",
    10: "violet",
    11: "magenta",
}


def train_visualize_random_forest(
    feature_matrix: np.array,
    labels: list,
    rf_estimators: int,
    rf_random_state: int = 42,
) -> dict:
    """ Trains and visualizes a random forest classifier.

    Notes:
        This isn't intended to be particularly flexible and and is more for demo purposes.

    Args:
        feature_matrix: (num_features x num_characters) matrix.
        labels: Ground-truth labels for the data set.
        rf_estimators: How many estimators should be used for the RF model?
        rf_random_state: Random state seed for reproducibility.

    Returns:
        A dictionary of character names, their predicted labels, and actual labels.
    """

    # Create the train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, random_state=rf_random_state
    )

    # Extract the character names from the test set to look at individual results later.
    char_names = X_test[:, 0]

    # Delete the character names column before feeding into model.
    X_train = np.delete(X_train, obj=0, axis=1)
    X_test = np.delete(X_test, obj=0, axis=1)

    # Train the random forest model.
    rfc = RandomForestClassifier(
        n_estimators=rf_estimators, random_state=rf_random_state
    ).fit(X_train, y_train)

    # Build the return dictionary
    labels_by_character = {}
    predicted_labels = rfc.predict(X_test)
    for idx, char in enumerate(char_names):
        ground_truth = y_test[idx]
        predicted = predicted_labels[idx]
        labels_by_character[char] = {
            "true_label": ground_truth,
            "predicted_label": predicted,
        }

    # Do the actual plotting
    fig, ax = plt.subplots(figsize=(20, 20))
    plot_confusion_matrix(rfc, X_test, y_test, normalize=None, ax=ax)
    plt.show()

    # Finally, return the dictionary of labels.
    return labels_by_character


def visualize_clustering_results(cluster_points: list, labels: list) -> None:
    """ Visualizes and labels the clusters resulting from an analysis.

    Args:
        cluster_points: [(x1, y1), (x2, y2), ..., (xN, yN)]
        labels: Label for each of the points in cluster_points.
    
    """

    # First, split out the point tuples by label.
    points_by_label = defaultdict(list)
    for idx, point in enumerate(cluster_points):
        points_by_label[labels[idx]].append(point)

    # Next, stack the points for each label into a single array.
    big_xy_list_by_label = {}
    for label, points_for_that_label in points_by_label.items():
        big_xy_list_by_label[label] = np.stack(tuple(points_for_that_label))

    # Compute the centroids of each point cloud for labeling.
    centroids_by_label = {}
    for label, arr in big_xy_list_by_label.items():
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        centroid = sum_x / length, sum_y / length
        centroids_by_label[label] = centroid

    # Initialize a counter to iterate through the color map
    i = 0
    plt.rcParams.update({"font.size": 22, "font.weight": "bold"})
    fig, ax = plt.subplots(figsize=(20, 20))
    for label, coords in centroids_by_label.items():
        ax.scatter(
            big_xy_list_by_label[label][:, 0],
            big_xy_list_by_label[label][:, 1],
            c=COLOR_DICT[i],
            s=50,
            alpha=0.5,
            label=label,
        )
        # plt.scatter(coords[0], coords[1], c=color_dict[i], label=label, s=100, alpha=0)
        ax.annotate(label, xy=coords, textcoords="data", color="black")
        i += 1
    ax.legend(loc="best")
    plt.show()


def tsne_points(feature_matrix: np.array, perplexity: int) -> list:
    """ Applies a t-SNE analysis and returns the character positions grouped by class.

    Notes:
        This isn't the cleanest, but I want parity with the way I analyze the UMAP data, so I
        chose the format to enable that. Because this is for plotting, I don't bother with more
        than 2 t-SNE dimensions (it's hard-coded), and I use 1000 iterations, which seems to offer
        well-converged results based on my testing. I use Manhattan distance for two reasons:

            1. The data is riddled with outliers in the feature space.
            2. The attribute features range from 0 - 20(ish) while the one-hot vectors are binary,
               and using Euclidean distance would weight the attributes too heavily.

    Args:
        feature_matrix: (num_features x num_characters) matrix.
        labels: Ground-truth labels for the data set.
        perplexity: Perplexity for the t-SNE model. N^(1/2) is a reasonable guess.

    Returns:
        A list of (x, y) tuples corresponding to the coordinates of each character in the embedding
        space.
    
    """
    number_of_t_sne_components = 2
    number_of_t_sne_iterations = 1000
    t_sne_metric = "manhattan"

    tsne = TSNE(
        n_components=number_of_t_sne_components,
        perplexity=perplexity,
        n_iter=number_of_t_sne_iterations,
        metric=t_sne_metric,
    )
    results = tsne.fit_transform(feature_matrix)

    # This is where my hacky plotting script makes us do unseemly things.
    tsne_1 = results[:, 0]
    tsne_2 = results[:, 1]
    plottable_list_form = []
    for idx in range(len(tsne_1)):
        plottable_list_form.append((tsne_1[idx], tsne_2[idx]))

    return plottable_list_form


def umap_points(
    feature_matrix: np.array, umap_neighors: int = 200, min_dist: float = 0.1
) -> list:
    """ As with the t-SNE method above, but with UMAP instead. 
    
    Notes:
        The choice of n_neighbors is currently defaulted to 200, because that's roughly the
        number of members of each class. min_dist was based on some empirical tuning.
    
    """
    mapper = umap.UMAP(n_neighbors=umap_neighors, min_dist=min_dist, metric="manhattan")
    u = mapper.fit_transform(feature_matrix)
    return list(u)
