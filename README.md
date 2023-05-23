# USER GAZE
source: https://github.com/ahmednull/l2cs-net

**TO RUN demo.py**
- python demo.py --snapshot .\models\L2CSNet_gaze360.pkl --cam 0

**Changes applied to make it run on CPU**:
- remove references to GPU-related functions or libraries such as "torch.backends.cudnn", "model.cuda()", "torch.FloatTensor().cuda()"
- replace these with corresponding CPU tensor "torch.FloatTensor(idx_tensor)
- Change device for model evaluation and data loading to CPU: "model.to("cpu")", "img.to("cpu")"
- Apply changes also to the detector the model uses.
  detector.py -> class RetinaFace: this class was built for GPU. It has been modified to make it work for CPU. Set gpu_id = -1.
  
  
**CARDINAL DIRECTION DETECTION** \
The *get_direction* function takes two arguments gaze_yaw and gaze_pitch, which are the horizontal and vertical angles of a person's gaze, respectively.

It first computes the destination point on a sphere using the horizontal and vertical angles of the gaze. Then, it calculates the angle between the destination point and the origin point (the center of the sphere) using math.atan2 function. The angle is converted from radians to degrees and stored in the variable angle.

Finally, based on the angle value, the function returns the direction of the gaze, which can be one of the four cardinal directions: W for west, E for east, N for north, or S for south. The angle ranges used to determine the direction are -45 to -135 for west, 45 to 135 for east, less than -135 or greater than or equal to 135 for north, and everything else for south.\


https://user-images.githubusercontent.com/92394378/236919329-42a0eea4-79d5-4e87-9449-ea4bda5fc9cc.mp4


### previous experiment (experimenting)
**FRAME CROPPING** \
This function takes as input a frame from a webcam or camera, along with gaze_yaw and gaze_pitch values, which are predicted by a gaze estimation algorithm. It also takes the crop_size of the region around the gaze point to be cropped and the save_folder, where the cropped images will be saved with a timestamp. Lastly, it takes the direction of the gaze point, which is predicted by the get_direction function.

The function first calculates the gaze coordinates on the frame by converting gaze_yaw and gaze_pitch values into pixel coordinates, which are based on the frame dimensions. Then, based on the direction of the gaze point, it calculates the crop region to be extracted from the frame.

If the direction is West, the function crops a square region of size crop_size around the left side of the gaze point. If the direction is East, the function crops a square region of size crop_size around the right side of the gaze point. If the direction is North, the function crops a square region of size crop_size around the top of the gaze point. If the direction is South, the function crops a square region of size crop_size around the bottom of the gaze point.

After calculating the crop region, the function crops the region from the frame and saves it to the save_folder with a timestamp. Finally, it returns the cropped frame as output.
The idea is to apply Detectron2 object detection algorithm to this cropped frame to identify the objectt the user is looking at. 

### new experiment
**GAZE AREA**:
The modified draw_gaze function, in utils.py, computes the gaze area by drawing a rectangle around the tip of the gaze arrow. The size of the rectangle can be controlled using the scale parameter. The function takes eye positions and gaze angles as inputs and visualizes the gaze angle on an image by drawing an arrow starting from the eye position and extending towards the gaze direction. Additionally, it includes a rectangle that represents the gaze area around the tip of the gaze arrow.


https://github.com/hikkaaa/WORK/assets/92394378/ada00a03-5b6b-47ef-9f79-9582a254a037



## TO DO:
- apply Detecron2 algorithm to cropped frame
- adjust crop size according to experiment environment setup
