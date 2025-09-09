import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def visualize_coco_predictions(predictions, image_dir, output_dir, json_file, alpha=0.1):
    """
    Visualize COCO-format predictions on all images.

    如果 predictions 中对于某张测试图片没有任何预测结果，就不会进行可视化绘制

    Args:
        predictions (list): List of predictions in COCO format.
        image_dir (str): Directory containing input images.
        output_dir (str): Directory to save the visualized images.
        json_file (str): Path to the COCO-format test.json file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the test.json file to map image_id to file_name
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # Map image_id to file_name
    image_id_to_filepath = {
        img['id']: os.path.join(image_dir, img['file_name'])
        for img in coco_data['images']
    }

    # Group predictions by image_id
    predictions_by_image = {}
    for prediction in predictions:
        image_id = prediction['image_id']
        if image_id not in predictions_by_image:
            predictions_by_image[image_id] = []
        predictions_by_image[image_id].append(prediction)

    for image_id, predictions in tqdm(predictions_by_image.items(), desc="Processing images"):
        if image_id not in image_id_to_filepath:
            print(f"Image ID {image_id} not found in the image directory.")
            continue

        # Read the image
        image_path = image_id_to_filepath[image_id]
        # print("image_path: ", image_path)  # e.g. ./data/enschede_road/test_images1/True_Ortho_2553_4734_patch_6.png
        image_name = os.path.basename(image_path)  # Extract the filename

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        overlay = np.zeros_like(image, dtype=np.uint8)

        # Draw all polygon transparent masks on the image
        for prediction in predictions:
            segmentation = prediction['segmentation']
            color = (0, 255, 0)  # Green for external polygons

            segmentation[0] = (np.array(segmentation[0]) * 2).tolist()  # 外部轮廓放大
            for i in range(1, len(segmentation)):
                segmentation[i] = (np.array(segmentation[i]) * 2).tolist()  # 内部轮廓放大

            # Draw external polygon
            exterior = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(overlay, [exterior], color)

            # Draw internal polygons (holes)
            for interior in segmentation[1:]:
                interior = np.array(interior, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(overlay, [interior], (0, 0, 0))

        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # Draw all polygon outlines and vertices on the image
        for prediction in predictions:
            segmentation = prediction['segmentation']
            color = (255, 255, 0)  # Cyan for external polygons
            vertex_color = (204, 102, 255)  # Rose pink for vertices

            # Draw external polygon
            exterior = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))

            cv2.polylines(image, [exterior], isClosed=True, color=color, thickness=1)

            # Draw vertices for the external polygon
            for vertex in exterior:
                cv2.circle(image, tuple(vertex), radius=3, color=vertex_color, thickness=-1)

            # Draw internal polygons (holes)
            for interior in segmentation[1:]:
                interior = np.array(interior, dtype=np.int32).reshape((-1, 2))
                cv2.polylines(image, [interior], isClosed=True, color=color, thickness=1)

                # Draw vertices for the internal polygon
                for vertex in interior:
                    cv2.circle(image, tuple(vertex), radius=3, color=vertex_color, thickness=-1)

        # Save the visualization
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        print("output_path: ", output_path)
        cv2.imwrite(output_path, image)


def visualize_single_image(image_path, predictions, output_path, alpha=0.1):
    """
    Visualize COCO-format predictions on a single image.

    Args:
        image_path (str): Path to the input image.
        predictions (list): List of predictions in COCO format.
        output_path (str): Path to save the visualized image.
    """
    # Extract image ID from filename
    image_id = int(os.path.splitext(os.path.basename(image_path))[0])

    # Filter predictions for the specified image_id
    predictions_for_image = [pred for pred in predictions if pred["image_id"] == image_id]

    # Read the image
    image = cv2.imread(image_path)

    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    overlay = np.zeros_like(image, dtype=np.uint8)

    # Draw all polygons on the image
    for prediction in predictions_for_image:
        segmentation = prediction['segmentation']
        color = (0, 255, 0)  # Cyan for external polygons

        segmentation[0] = (np.array(segmentation[0]) * 2).tolist()  # 外部轮廓放大
        for i in range(1, len(segmentation)):
            segmentation[i] = (np.array(segmentation[i]) * 2).tolist()  # 内部轮廓放大

        # Draw external polygon
        exterior = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))

        cv2.fillPoly(overlay, [exterior], color)

        # Draw internal polygons (holes)
        for interior in segmentation[1:]:
            interior = np.array(interior, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(overlay, [interior], (0, 0, 0))

    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw all polygons on the image
    for prediction in predictions_for_image:
        segmentation = prediction['segmentation']
        color = (255, 255, 0)  # Cyan
        vertex_color = (204, 102, 255)  # Rose pink

        # Draw external polygon
        exterior = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))

        cv2.polylines(image, [exterior], isClosed=True, color=color, thickness=2)

        # Draw vertices for the external polygon
        for vertex in exterior:
            cv2.circle(image, tuple(vertex), radius=4, color=vertex_color, thickness=-1)

        # Draw internal polygons (holes)
        for interior in segmentation[1:]:
            interior = np.array(interior, dtype=np.int32).reshape((-1, 2))
            cv2.polylines(image, [interior], isClosed=True, color=color, thickness=2)

            # Draw vertices for the internal polygon
            for vertex in interior:
                cv2.circle(image, tuple(vertex), radius=4, color=vertex_color, thickness=-1)

    # Save the visualization
    cv2.imwrite(output_path, image)


# Example usage
if __name__ == "__main__":
    predictions_path = "./outputs/vaihingen_map_generalization/epoch=epoch=59/polygons_seg_ddim_vertices_from_scaled1_heat_th-0.1_k-5.0_3.2_dp_eps2.json"  # Path to the COCO predictions JSON file
    image_directory = "./data/vaihingen_map_generalization/val/geb10_masks"  # Directory containing the input images
    json_file = "./data/vaihingen_map_generalization/geb10_val_annotations.json"
    output_directory = "./outputs/vaihingen_map_generalization/epoch=epoch=59/viz_ddim_vertices_from_scaled1_heat_th-0.1_k-5.0_3.2_dp_eps2"  # Directory to save visualized images

    # Load predictions from JSON
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    # Visualize all images
    visualize_coco_predictions(predictions, image_directory, output_directory, json_file, alpha=0.1)

    # single_image_path = ""  # Specify a single image to visualize
    # single_image_output_path = ""
    # Visualize a single image
    # visualize_single_image(single_image_path, predictions, single_image_output_path, alpha=0.1)


