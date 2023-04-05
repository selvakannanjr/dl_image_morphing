# Image Morphing

## Overview

* Find keypoints in image
* Implement the delaunay triangulation for the points
* Transform the image w.r.t the triangle

## Which dataset is used?
[300w](https://ibug.doc.ic.ac.uk/resources/300-W/) dataset

## How many kepoints are identified?
68 facial keypoints

## How to run the model?
```
git clone git@github.com:selvakannanjr/dl_image_morphing.git
cd face-morphing-multiple-images
python code/utils/align_images.py raw_images/ aligned_images/ --output_size=1024 # To align the images
python code/main.py --folder aligned_images --output video_output.mp4 --duration 4
```

## Sample Output




https://user-images.githubusercontent.com/26718058/230106604-857b677e-d577-4ce7-a0db-287ba4432f54.mp4

