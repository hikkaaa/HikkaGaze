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
