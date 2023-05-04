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
  
  **CARDINAL DIRECTION DETECTION* \
The *get_direction* function takes in two arguments gaze_yaw and gaze_pitch, which represent the yaw and pitch of the gaze direction, respectively. The yaw is the angle between the gaze direction and the horizontal axis, and the pitch is the angle between the gaze direction and the vertical axis. These angles are usually measured in radians.

The function first converts the angles from radians to degrees, as it is more common to work with degrees when dealing with directions. Then, it computes the destination point using the sin and cos trigonometric functions based on the yaw and pitch angles. The sin and cos functions return the x and y coordinates of the destination point on a unit sphere centered at the origin, respectively.

Next, the function calculates the angle between the destination point and the positive x-axis using the atan2 function. The atan2 function returns the angle in radians, so the result is converted to degrees using the multiplication factor of 180/pi.

Finally, the function determines the direction based on the angle. If the angle is between -45 and -135 degrees, the direction is considered to be west (W). If the angle is between 45 and 135 degrees, the direction is considered to be east (E). If the angle is between -135 and -45 degrees or between 135 and 45 degrees, the direction is considered to be north (N). Otherwise, the direction is considered to be south (S).
