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

def define_lanes_from_lines(lanes_with_ids, image_height, image_width):
    """
    Define complete lanes by pairing adjacent lane lines
    Calculate the center line for each lane
    Returns list of lane tuples: (lane_id, left_line, right_line, center_line)
    where each line is represented as (x1, y1, x2, y2)
    """
    if len(lanes_with_ids) < 2:
        return []
    
    # Extract line positions from lanes_with_ids
    line_positions = []
    for lane_id, x1, y1, x2, y2 in lanes_with_ids:
        mid_x = (x1 + x2) / 2
        line_positions.append((lane_id, mid_x, (x1, y1, x2, y2)))
    
    # Sort lines from left to right
    line_positions.sort(key=lambda x: x[1])
    
    # Pair adjacent lines to form lanes
    complete_lanes = []
    for i in range(len(line_positions) - 1):
        left_id, left_mid_x, left_line = line_positions[i]
        right_id, right_mid_x, right_line = line_positions[i+1]
        lane_id = i + 1  # New lane ID
        
        left_x1, left_y1, left_x2, left_y2 = left_line
        right_x1, right_y1, right_x2, right_y2 = right_line
        
        # Calculate center line coordinates
        center_x1 = int((left_x1 + right_x1) / 2)
        center_y1 = int((left_y1 + right_y1) / 2)
        center_x2 = int((left_x2 + right_x2) / 2)
        center_y2 = int((left_y2 + right_y2) / 2)
        
        # Store the lane information
        center_line = (center_x1, center_y1, center_x2, center_y2)
        complete_lanes.append((lane_id, left_line, right_line, center_line))
    
    return complete_lanes

def save_lane_data_to_csv(complete_lanes, lane_lines, image_file, output_folder):
    """
    Save lane data (IDs, line equations, center lines) to CSV files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get the image name without extension to use as part of the CSV filename
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    
    # Define the output CSV file paths
    lines_file = os.path.join(output_folder, f"{image_name}_lines.csv")
    lanes_file = os.path.join(output_folder, f"{image_name}_lanes.csv")

    # Save lane lines data
    with open(lines_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["line_id", "A", "B", "C", "x1", "y1", "x2", "y2"])
        for lane_id, x1, y1, x2, y2 in lane_lines:
            A, B, C = line_to_equation(x1, y1, x2, y2)
            writer.writerow([lane_id, A, B, C, x1, y1, x2, y2])
    
    # Save complete lanes data
    with open(lanes_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["lane_id", "left_line_id", "right_line_id", 
                         "center_x1", "center_y1", "center_x2", "center_y2",
                         "lane_width"])
        
        for lane_id, left_line, right_line, center_line in complete_lanes:
            left_id = lane_id  # Using indices from original lane lines
            right_id = lane_id + 1
            
            center_x1, center_y1, center_x2, center_y2 = center_line
            
            # Calculate average lane width
            left_x1, left_y1, left_x2, left_y2 = left_line
            right_x1, right_y1, right_x2, right_y2 = right_line
            width_top = abs(right_x1 - left_x1)
            width_bottom = abs(right_x2 - left_x2)
            avg_width = (width_top + width_bottom) / 2
            
            writer.writerow([lane_id, left_id, right_id, 
                            center_x1, center_y1, center_x2, center_y2, 
                            avg_width])
    
    return lines_file, lanes_file

def visualize_lanes_with_centers(image_path, lane_lines, complete_lanes, output_folder):
    """
    Visualize lane lines, complete lanes, and lane centers on the image
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Define colors for visualization
    line_color = (0, 0, 255)  # Red for lane lines
    center_color = (0, 255, 0)  # Green for lane centers
    
    # Draw lane lines
    for lane_id, x1, y1, x2, y2 in lane_lines:
        cv2.line(vis_image, (x1, y1), (x2, y2), line_color, 2)
        cv2.putText(vis_image, f"Line {lane_id}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
    
    # Draw lane centers
    for lane_id, _, _, center_line in complete_lanes:
        cx1, cy1, cx2, cy2 = center_line
        cv2.line(vis_image, (cx1, cy1), (cx2, cy2), center_color, 2)
        cv2.putText(vis_image, f"Lane {lane_id}", (cx1, cy1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, center_color, 2)
    
    # Create visualization folder if it doesn't exist
    vis_folder = os.path.join(output_folder, 'visualization')
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
    
    # Save the visualized image
    output_path = os.path.join(vis_folder, f"lanes_centers_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, vis_image)
    
    return output_path

def process_images_in_folder(folder_path, output_folder):
    """
    Process all images in a folder to detect lane lines, define lanes, and calculate lane centers
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
        
        # Assign unique IDs to lane lines
        lane_lines = assign_lane_ids(lane_positions, image_width)
        
        # Define complete lanes from pairs of lane lines
        complete_lanes = define_lanes_from_lines(lane_lines, image_height, image_width)
        
        # Save lane data to CSV files
        lines_file, lanes_file = save_lane_data_to_csv(complete_lanes, lane_lines, image_file, csv_folder)
        
        # Visualize lanes with IDs and centers
        vis_file = visualize_lanes_with_centers(image_path, lane_lines, complete_lanes, output_folder)
        
        print(f"Found {len(lane_lines)} lane lines and {len(complete_lanes)} complete lanes in {image_file}")
        print(f"Lane lines data saved to {lines_file}")
        print(f"Complete lanes data saved to {lanes_file}")
        print(f"Visualization saved to {vis_file}")

def main():
    """
    Main function to run the lane detection and center line calculation process
    """
    # Get the root directory of the script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Specify the folder containing images
    image_folder = os.path.join(root_dir, 'laneID-Eyelevel')  # Replace with your folder name
    output_folder = os.path.join(root_dir, 'output')
    
    # Process all images in the folder
    process_images_in_folder(image_folder, output_folder)
    
    print(f"Lane detection and center line calculation completed.")
    print(f"Visualization images saved to: {os.path.join(output_folder, 'visualization')}")
    print(f"CSV files saved to: {os.path.join(output_folder, 'csv_files')}")

if __name__ == "__main__":
    main()