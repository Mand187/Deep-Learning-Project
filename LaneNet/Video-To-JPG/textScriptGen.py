import os

# Define the path to your output frames directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "output_frames")

# Get a sorted list of all .jpg files in the directory
frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

# Define the path to the custom test file in the local folder
test_file_path = os.path.join(script_dir, 'custom_test.txt')

# Open the test file in write mode
with open(test_file_path, 'w') as test_file:
    for frame in frame_files:
        # Write the relative path of each frame to the test file
        test_file.write(f'output_frames/{frame}\n')

print(f'Custom test file created at {test_file_path}')
