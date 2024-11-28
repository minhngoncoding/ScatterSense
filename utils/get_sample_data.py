# Description: Check the training json file and extract the sample data from it

import json
import os
import cv2
from icecream import ic

NUM_DATA = 20

def extract_json_data(org_json_path):
    with open(org_json_path, "r") as f:
        data = json.load(f)

    sample_json = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    id_list = []

    for i in range(NUM_DATA):
        temp_image = data["images"][i]
        id_list.append(data["images"][i]["id"])
        sample_json["images"].append(temp_image)

    for i in data["annotations"]:
        if i["image_id"] in id_list:
            sample_json["annotations"].append(i)

    sample_json["categories"] = data["categories"]
    return sample_json

def get_image_in_json(json_file, image_dir, target_dir):
    import shutil
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    image_files = os.listdir(image_dir)
    with open(json_file, "r") as f:
        data = json.load(f)

    for image in data["images"]:
        image_name = image["file_name"]
        if image_name in image_files:
            src_path = os.path.join(image_dir, image_name)
            dst_path = os.path.join(target_dir, image_name)
            shutil.copy(src_path, dst_path)


def draw_line_key_points(json_file, image_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(json_file, "r") as f:
        data = json.load(f)

    for image_info in data["images"]:
        image_name = image_info["file_name"]
        image_id = image_info["id"]
        image_path = os.path.join(image_dir, image_name)
        ic(image_path, os.path.exists(image_path))

        image = cv2.imread(image_path)


        for annotation in data["annotations"]:
            if annotation["image_id"] == image_id:
                keypoints = annotation["bbox"]
                ic(keypoints)
                for i in range(0, len(keypoints), 2):
                    x, y= keypoints[i:i+2]
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(target_dir, image_name), image)
root_dir = "../data/linedata(1028)/line"
image_dir = os.path.join(root_dir, "sample_images")
target_dir = os.path.join(root_dir, "sample_images", "result")
json_file = os.path.join(root_dir, "sample_line_data.json")

draw_line_key_points(json_file, image_dir, target_dir)