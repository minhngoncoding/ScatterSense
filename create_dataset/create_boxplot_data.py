# Synthetic Vertical Box Plot Dataset Generator (COCO-style)
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from matplotlib import rcParams
from uuid import uuid4
from PIL import Image


class BoxplotDatasetGenerator:
    def __init__(
        self,
        num_images=5000,
        min_boxes=1,
        max_boxes=10,
        image_width=640,
        image_height=480,
        dpi=100,
        output_dir="data/verticalboxdata",
    ):
        self.num_images = num_images
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.image_width = image_width
        self.image_height = image_height
        self.dpi = dpi
        self.output_dir = output_dir
        self.img_dir = os.path.join(self.output_dir, "images")
        self.annotation_file = os.path.join(self.output_dir, "annotations.json")

        os.makedirs(self.img_dir, exist_ok=True)

        self.coco_output = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "Series", "supercategory": "Series"}],
        }
        self.annotation_id = 1

    def generate_random_label(self, length=10):
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=length))

    def generate_dataset(self):
        for img_id in tqdm(range(1, self.num_images + 1)):
            num_boxes = random.randint(self.min_boxes, self.max_boxes)
            data = [
                np.random.normal(
                    loc=random.uniform(10, 100), scale=random.uniform(5, 15), size=100
                )
                for _ in range(num_boxes)
            ]
            # Randomize image size
            random_width = random.randint(320, 1280)
            random_height = random.randint(240, 960)
            fig, ax = plt.subplots(
                figsize=(random_width / self.dpi, random_height / self.dpi),
                dpi=self.dpi,
            )

            ax.set_position((0.1, 0.1, 0.8, 0.8))
            ax.tick_params(
                axis="both",
                which="both",
                direction="out",
                length=5,
                width=1,
                colors="black",
                labelsize=8,
            )
            ax.set_title(self.generate_random_label(random.randint(10, 30)))
            ax.set_xlabel(self.generate_random_label(random.randint(5, 20)))
            ax.set_ylabel(self.generate_random_label(random.randint(5, 20)))
            plt.tight_layout(pad=0.5)

            use_default_aesthetic = random.random() < 0.1

            if use_default_aesthetic:
                bp = ax.boxplot(
                    data,
                    vert=True,
                    patch_artist=True,
                    flierprops=dict(
                        marker=random.choice(["x", "o", "s", "D", "^", "v"]),
                        color="red",
                        markersize=6,
                    ),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor("skyblue")
                    patch.set_edgecolor("black")
                    patch.set_linewidth(1.0)
                for whisker in bp["whiskers"]:
                    whisker.set_linestyle("-")
                    whisker.set_linewidth(1.0)
                for cap in bp["caps"]:
                    cap.set_linewidth(1.0)
                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.5)
            else:
                box_colors = random.choices(plt.cm.tab20.colors, k=num_boxes)
                box_linewidth = random.uniform(0.5, 2.5)
                box_widths = random.uniform(0.3, 0.9)

                if random.random() < 0.5:
                    box_colors = [box_colors[0]] * num_boxes

                bp = ax.boxplot(
                    data,
                    vert=True,
                    patch_artist=True,
                    widths=box_widths,
                    boxprops=dict(linewidth=box_linewidth),
                    flierprops=dict(
                        marker=random.choice(["x", "o", "s", "D", "^", "v"]),
                        color="red",
                        markersize=6,
                    ),
                )
                for patch, color, hatch in zip(
                    bp["boxes"],
                    box_colors,
                    ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"],
                ):
                    patch.set_facecolor(color)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(box_linewidth)
                    if random.random() < 0.1:
                        patch.set_hatch(hatch)

                for whisker in bp["whiskers"]:
                    whisker.set_color(random.choice(box_colors))
                    whisker.set_linewidth(random.uniform(0.5, 2.0))
                    whisker.set_linestyle(random.choice(["-", "--", "-.", ":"]))

                for cap in bp["caps"]:
                    cap.set_color(random.choice(box_colors))
                    cap.set_linewidth(random.uniform(0.5, 2.0))
                for flier in bp["fliers"]:
                    flier.set(
                        marker=random.choice(["o", "s", "D", "*", "+"]),
                        color=random.choice(box_colors),
                        alpha=random.uniform(0.6, 1.0),
                        markersize=random.uniform(5, 10),
                    )
                for median in bp["medians"]:
                    median.set_color(random.choice(plt.cm.tab10.colors))
                    median.set_linewidth(random.uniform(1.0, 2.0))

            ax.set_facecolor(random.choice(["white", "#f0f0f0", "#eaeaf2"]))
            if random.random() < 0.1:  # 10% chance for custom gridlines
                plt.grid(visible=True, linestyle="--", alpha=0.7)
            else:
                plt.grid(True if random.random() > 0.5 else False)

            file_name = f"{uuid4().hex}.png"
            file_path = os.path.join(self.img_dir, file_name)
            fig.canvas.draw()
            plt.savefig(file_path, dpi=self.dpi)
            image = Image.open(file_path)
            width, height = image.size

            self.coco_output["images"].append(
                {"id": img_id, "file_name": file_name, "width": width, "height": height}
            )

            for box_idx, d in enumerate(data):
                stats = [
                    np.min(d),
                    np.percentile(d, 25),
                    np.percentile(d, 50),
                    np.percentile(d, 75),
                    np.max(d),
                ]
                iqr = stats[3] - stats[1]
                lower_bound = stats[1] - 1.5 * iqr
                upper_bound = stats[3] + 1.5 * iqr
                # max_no_outliers = np.max(d[d < upper_bound])
                # min_no_outliers = np.min(d[d > lower_bound])
                # print(stats)
                # print(max_no_outliers)
                # print(min_no_outliers)
                stats[0] = max(lower_bound, np.min(d[d >= lower_bound]))
                stats[4] = min(upper_bound, np.max(d[d <= upper_bound]))

                flat_bbox = []

                for val in stats:
                    # for val in stats + [min_no_outliers, max_no_outliers]:
                    # if (val == stats[4] and val in flat_bbox) or (
                    #     val == stats[0] and val in flat_bbox
                    # ):
                    #     continue  # Avoid duplicate min or max values
                    x_disp, y_disp = ax.transData.transform((box_idx + 1, val))
                    _, image_height = fig.canvas.get_width_height()
                    x_img = x_disp
                    y_img = image_height - y_disp
                    flat_bbox.extend([round(x_img), round(y_img)])

                self.coco_output["annotations"].append(
                    {
                        "id": self.annotation_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "bbox": flat_bbox,
                        "area": 0,
                    }
                )
                self.annotation_id += 1

            plt.close()

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.annotation_file, "w") as f:
            json.dump(self.coco_output, f, indent=2)

        print(f"Dataset created at: {self.output_dir}")

    def visualize_sample(self, num_images=None):
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        images = data["images"][:num_images] if num_images else data["images"]
        for image_info in images:
            image_id = image_info["id"]
            image_path = os.path.join(self.img_dir, image_info["file_name"])
            annotations = [
                ann for ann in data["annotations"] if ann["image_id"] == image_id
            ]

            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            ax = plt.gca()

            for ann in annotations:
                bbox = ann["bbox"]
                xs = bbox[0::2]
                ys = bbox[1::2]
                ax.scatter(xs, ys, color="red", s=20, label=f"Box ID {ann['id']}")

            plt.title(f"Keypoints Visualization for Image ID {image_id}")
            plt.axis("off")
            plt.legend()
            save_path = os.path.join(
                viz_dir, f"visualization_{image_info['file_name']}.png"
            )
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Visualization saved to: {save_path}")


# Example usage:
if __name__ == "__main__":
    generator = BoxplotDatasetGenerator(num_images=10000)
    generator.generate_dataset()
    # generator.visualize_sample(1)
