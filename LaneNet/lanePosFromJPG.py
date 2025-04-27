import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from sklearn.cluster import DBSCAN
import math

def extract_lane_lines(image_path):
    """
    Extract red lane lines from an image and return their positions
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not read image: {image_path}")
        return []
        
    # Create a copy for visualization
    output_image = image.copy()
    
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color
    # Red wraps around in HSV, so we need two masks
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply Hough Line Transform to detect lines in the red mask
    edges = cv2.Canny(red_mask, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
    
    lane_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter out horizontal lines (less than 10 degrees from horizontal)
            if abs(y2 - y1) / (abs(x2 - x1) + 0.001) > 0.1:  # Adding 0.001 to avoid division by zero
                lane_positions.append((x1, y1, x2, y2))
    
    return lane_positions

def line_to_equation(x1, y1, x2, y2):
    """
    Convert a line defined by two points to the standard form Ax + By + C = 0
    """
    # Calculate the coefficients A, B, C of the line equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    # Normalize the coefficients
    norm = np.sqrt(A*A + B*B)
    if norm > 0:
        A, B, C = A/norm, B/norm, C/norm
    
    return A, B, C

def assign_lane_ids(lane_positions, image_width):
    """
    Assign unique IDs to lanes based on their position and slope
    Uses DBSCAN clustering to group similar lanes
    """
    if not lane_positions:
        return []
    
    # Calculate midpoints and slopes for each line segment
    features = []
    for x1, y1, x2, y2 in lane_positions:
        midpoint_x = (x1 + x2) / 2 / image_width  # Normalize by image width
        slope = math.atan2(y2 - y1, x2 - x1) if x2 != x1 else math.pi/2
        features.append([midpoint_x, slope])
    
    # Use DBSCAN to cluster similar lines
    features = np.array(features)
    db = DBSCAN(eps=0.1, min_samples=1).fit(features)
    labels = db.labels_
    
    # Combine lane positions with their assigned cluster IDs
    lanes_with_ids = []
    for i, (x1, y1, x2, y2) in enumerate(lane_positions):
        lane_id = int(labels[i]) + 1  # Add 1 to avoid 0-based indexing
        lanes_with_ids.append((lane_id, x1, y1, x2, y2))
    
    # Sort lanes from left to right based on midpoint x-coordinate
    lanes_with_ids.sort(key=lambda x: (x[1] + x[3])/2)
    
    # Reassign IDs based on left-to-right ordering
    final_lanes = []
    current_id = 1
    for lane_id, x1, y1, x2, y2 in lanes_with_ids:
        final_lanes.append((current_id, x1, y1, x2, y2))
        current_id += 1
    
    return final_lanes

def save_lane_data_to_individual_csv(lanes_with_ids, image_file, output_folder):
    """
    Save lane data (IDs and equations) to an individual CSV file for each image
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get the image name without extension to use as part of the CSV filename
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    
    # Define the output CSV file path
    output_file = os.path.join(output_folder, f"{image_name}_lanes.csv")

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header
        writer.writerow(["lane_id", "A", "B", "C", "x1", "y1", "x2", "y2"])

        # Write the lane data for the current image
        for lane_id, x1, y1, x2, y2 in lanes_with_ids:
            A, B, C = line_to_equation(x1, y1, x2, y2)
            writer.writerow([lane_id, A, B, C, x1, y1, x2, y2])
    
    return output_file

def process_images_in_folder(folder_path, output_folder):
    """
    Process all images in a folder to detect lane lines, assign IDs, and save data to individual CSV files
    """
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a CSV folder
    csv_folder = os.path.join(output_folder, 'csv_files')
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")
        
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue
            
        image_height, image_width = image.shape[:2]
        
        # Extract lane positions
        lane_positions = extract_lane_lines(image_path)
        
        # Assign unique IDs to lanes
        lanes_with_ids = assign_lane_ids(lane_positions, image_width)
        
        # Save lane data to an individual CSV file
        csv_file = save_lane_data_to_individual_csv(lanes_with_ids, image_file, csv_folder)
        
        # Visualize lanes with IDs
        visualize_lanes_with_ids(image_path, lanes_with_ids, output_folder)
        
        print(f"Found {len(lanes_with_ids)} unique lanes in {image_file}")
        print(f"Lane data saved to {csv_file}")

def visualize_lanes_with_ids(image_path, lanes_with_ids, output_folder):
    """
    Visualize lanes with their IDs on the image and save the result
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Draw lanes with different colors and add lane IDs
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark blue
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark red
        (128, 128, 0)   # Olive
    ]
    
    for lane_id, x1, y1, x2, y2 in lanes_with_ids:
        # Get color based on lane ID
        color = colors[(lane_id - 1) % len(colors)]
        
        # Draw the lane line
        cv2.line(image, (x1, y1), (x2, y2), color, 3)
        
        # Add lane ID text
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(image, f"Lane {lane_id}", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Create visualization folder if it doesn't exist
    vis_folder = os.path.join(output_folder, 'visualization')
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
    
    # Save the visualized image
    output_path = os.path.join(vis_folder, f"lanes_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, image)
    
    return output_path

def main():
    """
    Main function to run the lane detection and ID assignment process
    """
    # Get the root directory of the script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Specify the folder containing images
    image_folder = os.path.join(root_dir, 'laneID-Eyelevel')  # Replace with your folder name
    output_folder = os.path.join(root_dir, 'output')
    
    # Process all images in the folder
    process_images_in_folder(image_folder, output_folder)
    
    print(f"Lane detection and ID assignment completed.")
    print(f"Visualization images saved to: {os.path.join(output_folder, 'visualization')}")
    print(f"CSV files saved to: {os.path.join(output_folder, 'csv_files')}")

if __name__ == "__main__":
    main()