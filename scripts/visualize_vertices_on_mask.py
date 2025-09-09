import os
import json
import cv2
import numpy as np


def visualize_vertices_on_mask(vertices_json_path, mask_path, output_path, radius):
    """
    Visualize vertices on the mask (scaled if needed) and save the visualization.

    Args:
        vertices_json_path (str): Path to the JSON file containing extracted vertices.
        mask_path (str): Path to the mask file (binary mask).
        output_path (str): Directory to save the visualized images.
        sampler (str): Sampling method used (e.g., 'direct', 'ddim', 'ddpm').
    """
    # Load vertices from JSON file
    with open(vertices_json_path, 'r') as f:
        vertices_data = json.load(f)

    # Load mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(mask_path))[0]

    # Filter vertices corresponding to the mask
    vertices_info = next((item for item in vertices_data if file_name in item["image_file_name"]), None)
    if vertices_info is None:
        raise ValueError(f"No vertices found for the mask file: {file_name}")

    vertices = np.array(vertices_info["extracted_vertices"])
    if len(vertices) == 0:
        print(f"No vertices extracted for {file_name}. Skipping visualization.")
        return

    # Scale mask and vertices
    scale_factor = 2  # Scale by a factor of 2
    mask = cv2.resize(mask, (mask.shape[1] * scale_factor, mask.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST)
    vertices *= scale_factor  # Scale vertex coordinates

    # Convert mask to RGB for visualization
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw vertices on the mask
    for vertex in vertices:
        x, y = int(round(vertex[0])), int(round(vertex[1]))
        cv2.circle(mask_rgb, (x, y), radius=radius, color=(0, 0, 255), thickness=-1)  # Red dots for vertices

    # Save the visualized mask
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{file_name}_vertices.png")
    cv2.imwrite(output_file, mask_rgb)
    print(f"Visualization saved to: {output_file}")


def process_folder(vertices_json_path, mask_folder_path, output_path):
    """
    Process all mask images in a folder, visualize vertices on them, and save the results.

    Args:
        vertices_json_path (str): Path to the JSON file containing extracted vertices.
        mask_folder_path (str): Path to the folder containing mask images.
        output_path (str): Directory to save the visualized images.
        sampler (str): Sampling method used (e.g., 'direct', 'ddim', 'ddpm').
    """
    # List all image files in the mask folder
    mask_files = [os.path.join(mask_folder_path, f) for f in os.listdir(mask_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not mask_files:
        print(f"No valid mask images found in the folder: {mask_folder_path}")
        return

    # Process each mask file
    for mask_file in mask_files:
        try:
            visualize_vertices_on_mask(vertices_json_path, mask_file, output_path)
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")


if __name__ == "__main__":
    # vertices_json = ("./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/output_vertices_from_scaled1_heatmap_ddim_th-0.5_k-5.0_stitched.json")

    # mask_file = "./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/geb15_FTest1_pred.png"

    # output_dir = "./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79"

    # # Run the visualization on one image with scaling
    # visualize_vertices_on_mask(vertices_json, mask_file, output_dir)

    # Path to the folder containing mask files
    vertices_json = "./outputs/deventer_road/epoch=824-step=739199/output_vertices_from_scaled4_heatmap_ddim_th-0.1_k-3.0.json"

    mask_folder = ("./outputs/deventer_road/epoch=824-step=739199/samples_heat_ddim")

    output_dir = "./outputs/deventer_road/epoch=824-step=739199/output_vertices_on_heatmap"
    radius = 3

    # Run the visualization for all images in the folder
    process_folder(vertices_json, mask_folder, output_dir, radius)
