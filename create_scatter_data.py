from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
import cv2
from tqdm import tqdm
import math
from matplotlib.cm import get_cmap


ic.disable()

NUM_IMAGES = 25000
RANDOM_SIZE_PROB = 0.3
RANDOM_DIST_PROB = 0.3
RANDOM_MARKER_PROB = 0.3

MIN_DPI, MAX_DPI = 10, 200
MIN_MARKER_SIZE, MAX_MARKER_SIZE = 10, 200
MIN_NUM_POINTS, MAX_NUM_POINTS = 1, 500
MAX_NUM_MARKERS_TYPES = 4
COLOR_CHART_PROB = 0.3

ADDING_GRID_PROB = 0.1

DISTRIBUTION_TYPES= ["uniform", "linear", "quadratic", "sinusoidal", "exponential",
                     "logarithmic", "cubic", "circular", "elliptical", "spiral", "diamond", "cluster",
                     "threshold_scatter", "dense_vertical"]

def random_str(length=5):
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=length))

def create_colors_list():
    colors1 = list(plt.get_cmap('tab20').colors)
    colors2 = list(plt.get_cmap("viridis").colors)
    colors3 = list(plt.get_cmap("inferno").colors)

    colors = tuple(colors1 + colors2 + colors3)
    return colors

def generate_cluster_points(num_points, num_clusters=3, spread=1.0, order_of_magnitude=1):
    # Define cluster centers
    cluster_centers = [
        (np.random.uniform(0, 10 ** order_of_magnitude), np.random.uniform(0, 10 ** order_of_magnitude))
        for _ in range(num_clusters)
    ]

    # Assign each point to a cluster
    points_per_cluster = [num_points // num_clusters] * num_clusters
    points_per_cluster[-1] += num_points % num_clusters  # Handle remainder

    x, y = [], []
    for i, (cx, cy) in enumerate(cluster_centers):
        x.extend(np.random.normal(loc=cx, scale=spread, size=points_per_cluster[i]))
        y.extend(np.random.normal(loc=cy, scale=spread, size=points_per_cluster[i]))

    return np.array(x), np.array(y)


# Helper function to generate data points
def generate_data_points(num_points, dist_type="uniform", order_of_magnitude=1):
    distributions = {
        "uniform": lambda: (np.random.rand(num_points) * (10 ** order_of_magnitude),
                            np.random.rand(num_points) * (10 ** order_of_magnitude)),
        "linear": lambda: (np.linspace(0, 10 ** order_of_magnitude, num_points),
                           np.linspace(0, 10 ** order_of_magnitude, num_points) + np.random.normal(0, (10 ** (order_of_magnitude - 1)), num_points)),
        "quadratic": lambda: (np.linspace(0, 10 ** order_of_magnitude, num_points),
                              np.linspace(0, 10 ** order_of_magnitude, num_points) ** 2 / (10 ** (2 * order_of_magnitude)) + np.random.normal(0, 0.1, num_points)),
        "sinusoidal": lambda: (np.linspace(0, 4 * np.pi, num_points),
                               np.sin(np.linspace(0, 4 * np.pi, num_points)) * (10 ** order_of_magnitude) + np.random.normal(0, 0.1, num_points)),
        "exponential": lambda: (np.linspace(0, 10 ** order_of_magnitude, num_points),
                                np.exp(np.linspace(0, 10 ** order_of_magnitude, num_points) / (10 ** order_of_magnitude)) + np.random.normal(0, 0.1, num_points)),
        "logarithmic": lambda: (np.linspace(0.1, 10 ** order_of_magnitude, num_points),
                                np.log(np.linspace(0.1, 10 ** order_of_magnitude, num_points)) * (10 ** order_of_magnitude) + np.random.normal(0, 0.1, num_points)),
        "cubic": lambda: (np.linspace(0, 10 ** order_of_magnitude, num_points),
                          (np.linspace(0, 10 ** order_of_magnitude, num_points) ** 3) / (10 ** (3 * order_of_magnitude)) + np.random.normal(0, 0.1, num_points)),
        "circular": lambda: (
            np.cos(np.linspace(0, 2 * np.pi, num_points)) * 10 ** order_of_magnitude + np.random.normal(0, 0.1, num_points),
            np.sin(np.linspace(0, 2 * np.pi, num_points)) * 10 ** order_of_magnitude + np.random.normal(0, 0.1, num_points)),
        "elliptical": lambda: (
            np.cos(np.linspace(0, 2 * np.pi, num_points)) * 10 ** order_of_magnitude + np.random.normal(0, 0.1, num_points),
            np.sin(np.linspace(0, 2 * np.pi, num_points)) * (10 ** order_of_magnitude / 2) + np.random.normal(0, 0.1, num_points)),
        "spiral": lambda: (
            np.linspace(0, 2 * np.pi, num_points) * np.cos(np.linspace(0, 2 * np.pi, num_points)) * 10 ** order_of_magnitude,
            np.linspace(0, 2 * np.pi, num_points) * np.sin(np.linspace(0, 2 * np.pi, num_points)) * 10 ** order_of_magnitude),
        "diamond": lambda: (
            np.random.choice([-1, 1], num_points) * (np.random.rand(num_points) * 10 ** order_of_magnitude),
            np.random.choice([-1, 1], num_points) * (np.random.rand(num_points) * 10 ** order_of_magnitude)),
        "random": lambda: (np.random.rand(num_points) * (10 ** order_of_magnitude),
                           np.random.rand(num_points) * (10 ** order_of_magnitude)),
        "threshold_scatter": lambda: (np.random.normal(0, 5, num_points),  # Scatter with quadrants
                                      np.random.normal(0, 5, num_points)),
        "dense_vertical": lambda: (np.random.uniform(-5, 5, num_points),  # Dense vertical scatter
                                   np.random.normal(0, 2, num_points)),
        "cluster": lambda: generate_cluster_points(num_points, num_clusters=random.randint(1,5), spread=1.0,
                                                   order_of_magnitude=order_of_magnitude)
    }

    return distributions[dist_type]()


def map_data_to_image_coordinates(x, y, ax, fig, dpi, radius_pixels):
    """
    Maps data coordinates to image coordinates.

    Parameters:
        x, y: Data coordinates (arrays)
        ax: Axes object of the plot
        fig: Figure object of the plot


    Returns:
        List of (image_x, image_y) tuples in pixel coordinates
    """

    # Convert data coordinates to pixel coordinates

    fig_width, fig_height = fig.get_size_inches() * dpi
    image_width = int(fig_width)
    image_height = int(fig_height)

    # Get the axes bounding box in figure-relative coordinates
    bbox = ax.get_position()
    axes_x0, axes_y0 = bbox.x0 * image_width , bbox.y0 * image_height
    axes_x1, axes_y1 = bbox.x1 * image_width , bbox.y1 * image_height
    # Calculate the axes' width and height in pixels
    axes_width = axes_x1 - axes_x0
    axes_height = axes_y1 - axes_y0

    # Get the data limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Map data coordinates to pixel coordinates
    image_coords = []
    areas = []
    for data_x, data_y, radius in zip(x, y, radius_pixels):
        # Map data coordinates to axes-relative pixel coordinates
        pixel_x = axes_x0 + (data_x - xlim[0]) / (xlim[1] - xlim[0]) * axes_width
        pixel_y = axes_y0 + axes_height - (data_y - ylim[0]) / (ylim[1] - ylim[0]) * axes_height

        # Adjust for marker size (convert size in pt² to radius in pixels)
        pixel_y = math.ceil(pixel_y + radius//2)

        # Adjust pixel_y by marker radius to align to the center

        areas.append(math.ceil(radius))
        pixel_x, pixel_y = round(pixel_x), round(pixel_y)
        image_coords.extend([pixel_x, pixel_y])

    return image_coords, areas


def generate_scatter_plot_with_metadata(image_name, image_id):
    global current_id

    # Randomize number of points and distribution type
    num_points = random.randint(MIN_NUM_POINTS, MAX_NUM_POINTS)
    dist_type = random.choice(DISTRIBUTION_TYPES) if np.random.rand() > RANDOM_DIST_PROB else "random"
    x, y = generate_data_points(num_points, dist_type)

    # Randomize marker types and split points into groups
    marker_styles = ['.', ',', 'o', 'v', '^', '<', '>', 's', '*', 'x']
    use_mixed_markers = np.random.rand() <= RANDOM_MARKER_PROB
    if use_mixed_markers:
        num_groups = min(MAX_NUM_MARKERS_TYPES, random.randint(2, len(marker_styles)))
        # Shuffle all indices
        indices = np.random.permutation(num_points) if np.random.random() < 0.9 else np.arange(num_points)
        indices = np.array_split(indices, num_groups)
        markers = random.sample(marker_styles, num_groups)
        group_colors = random.sample(colors, num_groups)
    else:
        markers = [random.choice(marker_styles)]
        indices = [np.arange(num_points)]
        group_colors = [random.choice(colors)]
    # Determine if the entire image will use random sizes or consistent sizes
    random_sizes_per_image = random.random() <= RANDOM_SIZE_PROB

    # Initialize metadata list
    image_metadata = []
    plot_metadata = []

    # Plot and save data
    fig_width = random.uniform(6, 12)
    fig_height = random.uniform(4, 8)
    dpi = random.randint(MIN_DPI, MAX_DPI)


    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    use_black_and_white = np.random.rand() < COLOR_CHART_PROB
    if use_black_and_white:
        figure_bg_color = random.choice(["lightblue", "beige", "lavender", "mistyrose", "lightgray", "white",  'silver'])
        axes_bg_color = random.choice(["white", "lightyellow", "aliceblue", "honeydew", "lightcyan", "ivory",'darkgrey'])
    else:
        figure_bg_color = "white"
        axes_bg_color = "white"

    fig.patch.set_facecolor(figure_bg_color)  # Set figure background
    if np.random.rand() <= ADDING_GRID_PROB:
        major_linestyle = random.choice(["-", '--', '-.', ':'])
        major_linewidth = random.uniform(0.3, 1)
        ax.grid(True, color='black', linestyle=major_linestyle, linewidth=major_linewidth)
        if np.random.rand() <= ADDING_GRID_PROB:
            ax.minorticks_on()
            minor_linestyle = random.choice(["-", '--', '-.', ':'])
            minor_linewidth = random.uniform(0.2, 1)
            # Minor grid
            ax.grid(True, which='minor', color='gray', linestyle=minor_linestyle, linewidth=minor_linewidth)

    ax.set_facecolor(axes_bg_color)
    ax.set_title(random_str(random.randint(10, 30)))
    ax.set_xlabel(random_str(random.randint(5, 20)))
    ax.set_ylabel(random_str(random.randint(5, 20)))
    image_width, image_height = fig.get_size_inches() * dpi

    plot_data = []
    for i, (group, marker, color) in enumerate(zip(indices, markers, group_colors)):
        # Randomize marker sizes
        if random_sizes_per_image:
            size = random.randint(MIN_MARKER_SIZE, MAX_MARKER_SIZE)
            sizes = [size] * np.random.rand(len(group))
        else:
            sizes = [random.randint(MIN_MARKER_SIZE, MAX_MARKER_SIZE)] * len(group)

        if use_black_and_white:
            scatter = ax.scatter(x[group], y[group], s=sizes, marker=marker,color=color, alpha=random.uniform(0.7, 1.0))
        else:
            scatter = ax.scatter(x[group], y[group], marker=marker, color="black", alpha=random.uniform(0.7, 1.0))

        plot_data.append({
            "group": group,
            "sizes": sizes,
            "marker": marker,
        })

    for data in plot_data:
        group = data["group"]
        sizes = data["sizes"]
        marker = data["marker"]
        # Adjust for marker size (convert size in pt² to radius in pixels)

        radius_in_points = np.sqrt(sizes) / 2  # Convert size to radius (points)
        radius_in_pixels = radius_in_points * dpi / 72  # 1 point = 1/72 inch


        # This make sure that the keypoints is in the center
        # if marker in ["<", ">", "x"]:
        #     radius_in_pixels *= 1.4
        # Convert x, y to pixel coordinates using ax.transData
        bbox, areas = map_data_to_image_coordinates(x[group], y[group], ax, fig, dpi, radius_in_pixels)
        # Append plot metadata
        plot_metadata.append({
            "image_id": image_id,
            "category_id": 0,
            "bbox": bbox,  # Grouped bbox for all points with the same marker, in pixels
            "area": areas,  # Marker areas in pixels
            "id": current_id  # Unique ID for each marker type
        })

        current_id += 1

    image_path = os.path.join(output_dir, image_name)

    plt.savefig(image_path, dpi=dpi)
    # cv2.imshow(image_name, cv2.imread(image_path))
    # cv2.waitKey(0)

    # Append Image metadata
    image_metadata.append({
        "file_name": image_name,
        "height": int(image_height),
        "width": int(image_width),
        "id": image_id
    })

    plt.close()
    ic("===========================================")
    return image_metadata, plot_metadata


if __name__ == '__main__':
    # Generate plots and metadata
    metadata_list = {"images": [],
                     "annotations": [],
                     "categories": []}

    metadata_list["categories"].append(
        {
            "supercategory": "Series",
            "id": 0,
            "name": "Series"
        }
    )

    colors = create_colors_list()
    image_id = 0

    output_dir = "utils/simple_scatter_25000"
    os.makedirs(output_dir, exist_ok=True)

    global current_id

    current_id = 0

    for i in tqdm(range(NUM_IMAGES)):
        image_name = f"scatter_plot_{i}.png"
        image_metadata, plot_metadata = generate_scatter_plot_with_metadata(image_name, image_id)
        image_id += 1
        metadata_list["images"].extend(image_metadata)
        metadata_list["annotations"].extend(plot_metadata)

    # Save metadata to JSON
    metadata_file = os.path.join(output_dir, "scatter_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Generated {NUM_IMAGES} plots with pixel-based metadata saved to '{metadata_file}'.")

