## Description

The `lanePosFromJPG.py` script processes annotated images from the CHD dataset to extract lane line data. Currently, it outputs lane lines but does not provide full lane position information. The script relies on the red line annotations present in the CHD dataset.

## Usage

1. Place the annotated images from the CHD dataset into either the `laneId-Eyelevel` or `laneID-HighAngle` folder.
2. Run the script. It will automatically detect the images in the specified folder and split the lane line data into multiple CSV files.

Ensure the folder structure and annotations are correctly set up before running the script.