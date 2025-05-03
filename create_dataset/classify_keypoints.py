import numpy as np


def classify_keypoints_with_outliers(keypoints):
    # Sort keypoints by Y-coordinate (ascending)
    keypoints_sorted = sorted(keypoints, key=lambda kp: kp[1])

    # Group keypoints with the same Y-coordinate
    grouped_keypoints = {}
    for kp in keypoints_sorted:
        grouped_keypoints.setdefault(kp[1], []).append(kp)

    # Calculate vertical gaps between unique Y-coordinates
    unique_y_coords = sorted(grouped_keypoints.keys())
    gaps = np.diff(unique_y_coords)

    # Identify large gaps (e.g., using a threshold or clustering)
    threshold = np.mean(gaps) + 2 * np.std(gaps)  # Example threshold
    main_y_indices = [0]  # Start with the first Y-coordinate
    for i, gap in enumerate(gaps):
        if gap < threshold:
            main_y_indices.append(i + 1)

    # Separate main keypoints and outliers
    main_keypoints = [
        kp for i in main_y_indices for kp in grouped_keypoints[unique_y_coords[i]]
    ]
    outliers = [kp for kp in keypoints_sorted if kp not in main_keypoints]

    # Assign main keypoints (Min, Q1, Median, Q3, Max)
    if len(main_keypoints) >= 5:
        min_point = main_keypoints[0]
        q1_point = main_keypoints[1]
        median_point = main_keypoints[2]
        q3_point = main_keypoints[3]
        max_point = main_keypoints[4]
    else:
        raise ValueError("Not enough keypoints detected to classify main points.")

    return {
        "min": min_point,
        "q1": q1_point,
        "median": median_point,
        "q3": q3_point,
        "max": max_point,
        "outliers": outliers,
    }


if __name__ == "__main__":
    # Test case: Keypoints on the same vertical line
    keypoints = [
        (140, 265),
        (140, 170),
        (140, 151),
        (140, 129),
        (140, 47),
        (140, 226),
        (140, 76),
    ]

    result = classify_keypoints_with_outliers(keypoints)
    print("Test Result:", result)
    result = classify_keypoints_with_outliers(keypoints)

    print("Min:", result["min"])
    print("Q1:", result["q1"])
    print("Median:", result["median"])
    print("Q3:", result["q3"])
    print("Max:", result["max"])
    print("Outliers:", result["outliers"])
