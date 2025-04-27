import torch

# Lane-aware loss function
def lane_loss(predicted, lane_positions):
    distance_to_lane = 0
    for pred_point in predicted:
        min_distance = float('inf')
        for lane in lane_positions:
            A, B, C = lane  # Line equation coefficients
            x, y = pred_point
            # Calculate the perpendicular distance to the line Ax + By + C = 0
            distance = abs(A * x + B * y + C) / torch.sqrt(A ** 2 + B ** 2)
            min_distance = min(min_distance, distance)
        distance_to_lane += min_distance
    
    return distance_to_lane / len(predicted)