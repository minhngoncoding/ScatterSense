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

NUM_IMAGES = 5000
RANDOM_SIZE_PROB = 0.4
RANDOM_DIST_PROB = 0.3
RANDOM_MARKER_PROB = 0.8

MIN_DPI, MAX_DPI = 30, 300
MIN_MARKER_SIZE, MAX_MARKER_SIZE = 10, 500
MIN_NUM_POINTS, MAX_NUM_POINTS = 100, 600
MAX_NUM_MARKERS_TYPES = 10
DISTRIBUTION_TYPES= ["uniform", "linear", "quadratic", "sinusoidal", "exponential",
                     "logarithmic", "cubic", "circular", "elliptical", "spiral", "diamond"]

colors = plt.get_cmap('tab10').colors
# Directory to save plots and metadata
output_dir = "utils/complex_scatter_data1"
os.makedirs(output_dir, exist_ok=True)

global current_id
current_id = 85337

def random_str(length=5):
    return ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=length))


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
                           np.random.rand(num_points) * (10 ** order_of_magnitude))
    }
    return distributions[dist_type]()


def map_data_to_image_coordinates(x, y, ax, fig, dpi, marker_sizes):
    """
    Maps data coordinates to image coordinates.

    Parameters:
        x, y: Data coordinates (arrays)
        ax: Axes object of the plot
        fig: Figure object of the plot
        dpi: DPI of the figure

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
    for data_x, data_y, size in zip(x, y, marker_sizes):
        # Map data coordinates to axes-relative pixel coordinates
        pixel_x = axes_x0 + (data_x - xlim[0]) / (xlim[1] - xlim[0]) * axes_width
        pixel_y = axes_y0 + axes_height - (data_y - ylim[0]) / (ylim[1] - ylim[0]) * axes_height

        # Adjust for marker size (convert size in pt² to radius in pixels)
        radius_in_points = np.sqrt(size) / 2  # Convert size to radius (points)
        radius_in_pixels = radius_in_points * dpi / 72  # 1 point = 1/72 inch


        pixel_x, pixel_y = round(pixel_x), round(pixel_y)
        pixel_x = round(pixel_x)
        pixel_y = math.ceil(pixel_y + radius_in_pixels)


        # Adjust pixel_y by marker radius to align to the center
        ic(data_x, data_y)
        ic(xlim, ylim, axes_x0, axes_y0, axes_width, axes_height)
        ic(pixel_x, pixel_y)

        areas.append(math.ceil(radius_in_pixels))
        image_coords.extend([pixel_x, pixel_y])

    return image_coords, areas


def generate_scatter_plot_with_metadata(image_name, image_id):
    global current_id
    ic("Image Name:", image_name)
    
    # Randomize number of points and distribution type
    num_points = random.randint(MIN_NUM_POINTS, MAX_NUM_POINTS)
    if np.random.rand() > RANDOM_DIST_PROB:
        dist_type = random.choice(DISTRIBUTION_TYPES)
    else:
        dist_type = "random"
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

    figure_bg_color = random.choice(["lightblue", "beige", "lavender", "mistyrose", "lightgray", "white",  'silver'])
    axes_bg_color = random.choice(["white", "lightyellow", "aliceblue", "honeydew", "lightcyan", "ivory", "black",'darkgrey'])

    fig.patch.set_facecolor(figure_bg_color)  # Set figure background
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

        scatter = ax.scatter(x[group], y[group], s=sizes, marker=marker,color=color, alpha=random.uniform(0.7, 1.0))
        plot_data.append({
            "group": group,
            "sizes": sizes,
        })

    for data in plot_data:
        group = data["group"]
        sizes = data["sizes"]

        # Convert x, y to pixel coordinates using ax.transData
        bbox, areas = map_data_to_image_coordinates(x[group], y[group], ax, fig, dpi, sizes)
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

image_id = 25000
for i in tqdm(range(NUM_IMAGES)):
    image_name = f"scatter_plot_{i+25000}.png"
    image_metadata, plot_metadata = generate_scatter_plot_with_metadata(image_name, image_id)
    image_id += 1
    metadata_list["images"].extend(image_metadata)
    metadata_list["annotations"].extend(plot_metadata)

# Save metadata to JSON
metadata_file = os.path.join(output_dir, "scatter_metadata.json")
with open(metadata_file, "w") as f:
    json.dump(metadata_list, f, indent=4)

print(f"Generated {NUM_IMAGES} plots with pixel-based metadata saved to '{metadata_file}'.")

