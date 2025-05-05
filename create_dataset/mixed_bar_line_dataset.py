import os
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from uuid import uuid4
from tqdm import tqdm
from PIL import Image


class MixedBarLineDataset:
    def __init__(
        self,
        output_dir="data/mixeddata/mixed",
        num_charts=50,
        fig_size_range=(4, 12),
    ):
        self.output_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        self.visualized_dir = os.path.join(output_dir, "visualization")
        self.annotation_dir = os.path.join(output_dir, "annotations")
        self.annotation_file = os.path.join(self.annotation_dir, "annotations.json")
        self.num_charts = num_charts
        self.fig_size_range = fig_size_range

        # Create necessary directories
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.visualized_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)

        # Initialize dataset structure
        self.dataset = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "bar"}, {"id": 2, "name": "line"}],
        }

    def generate_data(self):
        """Generate random data for bar and line charts."""
        x = np.arange(1, random.randint(2, 15))  # Random x range
        bar_values = np.random.randint(20, 100, size=len(x))
        line_values = bar_values + np.random.randint(-20, 20, size=len(x))
        return x, bar_values, line_values

    def apply_random_style(self):
        """Apply random styles to the chart."""
        plt.style.use(random.choice(plt.style.available))
        plt.grid(random.choice([True, False]))
        plt.title(
            "".join(random.choices(string.ascii_letters, k=10)),
            fontsize=random.randint(10, 20),
        )
        plt.xlabel(
            "".join(random.choices(string.ascii_letters, k=6)),
            fontsize=random.randint(8, 15),
        )
        plt.ylabel(
            "".join(random.choices(string.ascii_letters, k=6)),
            fontsize=random.randint(8, 15),
        )

    def generate_chart(self, chart_id):
        # Generate data
        x, bar_values, line_values = self.generate_data()

        # Reset matplotlib style state to avoid cumulative side effects
        import matplotlib as mpl

        mpl.rcParams.update(mpl.rcParamsDefault)

        # Apply visual style
        self.apply_random_style()

        # Set figure size and DPI
        fig_size = (
            random.uniform(*self.fig_size_range),
            random.uniform(*self.fig_size_range),
        )
        dpi = 100  # consistent dpi across transform & save
        fig, ax = plt.subplots(figsize=fig_size)
        fig.set_dpi(dpi)

        # Calculate image pixel size
        width = int(fig_size[0] * dpi)
        height = int(fig_size[1] * dpi)

        # Plot bars
        bar_width = random.uniform(0.4, 0.8)
        bar_color = random.choice(
            [
                "skyblue",
                "blue",
                "green",
                "orange",
                "purple",
                "magenta",
                "cyan",
                "red",
                "gold",
                "teal",
            ]
        )

        bar_alpha = random.uniform(0.5, 1)
        bars = ax.bar(
            x,
            bar_values,
            width=bar_width,
            color=bar_color,
            alpha=bar_alpha,
            label="Bar Data",
        )

        # Plot line
        line_color = random.choice(
            [
                "red",
                "black",
                "cyan",
                "blue",
                "green",
                "magenta",
                "orange",
                "purple",
                "brown",
                "pink",
            ]
        )

        line_style = random.choice(
            ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 2, 3, 2))]
        )

        line_marker = random.choice(
            ["o", "s", "D", "^", "v", "<", ">", "p", "h", "*", "x", "+", ".", "|", "_"]
        )
        line = ax.plot(
            x,
            line_values,
            color=line_color,
            linestyle=line_style,
            marker=line_marker,
            label="Line Data",
        )

        # Clamp axes limits
        ax.set_xlim(left=0, right=max(x) + 1)
        ax.set_ylim(0, max(max(bar_values), max(line_values)) * 1.2)

        # Add legend
        ax.legend()

        fig.canvas.draw()

        # Save image
        image_file = os.path.join(self.image_dir, f"chart_{chart_id}.png")
        plt.tight_layout()
        plt.savefig(image_file, dpi=dpi)

        # Save visualized chart (optional)
        visualized_file = os.path.join(
            self.visualized_dir, f"chart_{chart_id}_visualized.png"
        )
        plt.savefig(visualized_file, dpi=dpi)

        # Store metadata
        self.dataset["images"].append(
            {
                "id": chart_id,
                "file_name": f"chart_{chart_id}.png",
                "width": width,
                "height": height,
            }
        )

        # Generate annotations AFTER draw
        self.generate_annotations(
            chart_id, x, bar_values, line_values, bar_width, fig, ax, bars, line
        )

        # ✅ Always close the specific figure to avoid memory leaks or state sharing
        plt.close(fig)

    def generate_annotations(
        self, image_id, x, bar_values, line_values, bar_width, fig, ax, bars, line
    ):
        annotation_id = len(self.dataset["annotations"]) + 1

        # Get image size in pixels
        fig_width, fig_height = fig.get_size_inches()
        fig_dpi = fig.get_dpi()
        img_width, img_height = int(fig_width * fig_dpi), int(fig_height * fig_dpi)

        # === BAR ANNOTATIONS ===
        for i, bar in enumerate(bars):
            x_topleft = bar.get_x()
            x_bottomright = x_topleft + bar.get_width()
            y_topleft = bar.get_height()
            y_bottomright = 0

            # Convert to image coordinates
            x1_disp, y1_disp = ax.transData.transform((x_topleft, y_topleft))
            x2_disp, y2_disp = ax.transData.transform((x_bottomright, y_bottomright))

            x1_img = int(x1_disp)
            y1_img = int(img_height - y1_disp)
            x2_img = int(x2_disp)
            y2_img = int(img_height - y2_disp)

            bbox = [x1_img, y1_img, x2_img, y2_img]
            area = (x2_img - x1_img) * (y2_img - y1_img)

            self.dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Bar
                    "bbox": bbox,
                    "area": area,
                }
            )
            annotation_id += 1

        # === LINE ANNOTATIONS ===
        line_coords = line[0].get_xydata()  # Extract (x, y) pairs

        keypoints = []
        for x_data, y_data in line_coords:
            x_disp, y_disp = ax.transData.transform((x_data, y_data))
            x_img = int(x_disp)
            y_img = int(img_height - y_disp)
            keypoints.extend([x_img, y_img])

        self.dataset["annotations"].append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2,  # Line
                "bbox": keypoints,
                "area": 0,
            }
        )

    def generate_dataset(self):
        """Generate the entire dataset."""
        for chart_id in tqdm(range(1, self.num_charts + 1)):
            self.generate_chart(chart_id)

        # Save annotations to JSON
        with open(self.annotation_file, "w") as f:
            json.dump(self.dataset, f, indent=4)
        print(f"Annotations saved to {self.annotation_file}")

    def visualize_sample(self, num_images=None):
        """Visualize keypoints based on the annotations file and save to visualization folder."""
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        images = data["images"][:num_images] if num_images else data["images"]

        for image_info in tqdm(images):
            image_id = image_info["id"]
            image_path = os.path.join(self.image_dir, image_info["file_name"])

            annotations = [
                ann for ann in data["annotations"] if ann["image_id"] == image_id
            ]

            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            ax = plt.gca()

            for ann in annotations:
                category_id = ann["category_id"]
                bbox = ann["bbox"]

                if category_id == 1:  # Bar
                    x_topleft, y_topleft, x_bottomright, y_bottomright = bbox
                    ax.scatter(x_topleft, y_topleft, color="yellow", s=50)
                    ax.scatter(x_bottomright, y_bottomright, color="blue")
                elif category_id == 2:  # Line
                    keypoints = (
                        bbox  # Assuming bbox contains [x1, y1, x2, y2, ..., xn, yn]
                    )
                    for i in range(0, len(keypoints), 2):
                        xi, yi = keypoints[i], keypoints[i + 1]
                        ax.scatter(xi, yi, color="red", s=50)
            plt.axis("off")

            # Save the visualized image
            visualized_file = os.path.join(
                self.visualized_dir, f"chart_{image_id}_visualized.png"
            )
            plt.savefig(visualized_file, bbox_inches="tight", pad_inches=0)
            plt.close()


if __name__ == "__main__":
    dataset = MixedBarLineDataset(num_charts=25000)
    dataset.generate_dataset()
    # dataset.visualize_sample(num_images=100)
