# WORK

https://www.youtube.com/watch?v=Pb3opEFP94U&t=780s


Objectron sources:
https://google.github.io/mediapipe/solutions/objectron.html#resources
https://colab.research.google.com/drive/1V7UxCaOuyeUWL2kwEc1RqgJrbtutzfPI?usp=sharing
https://ai.googleblog.com/2020/03/real-time-3d-object-detection-on-mobile.html


# USER GAZE
source: https://github.com/ahmednull/l2cs-net

Changes applied to make it run on CPU:
- remove references to GPU-related functions or libraries such as "torch.backends.cudnn", "model.cuda()", "torch.FloatTensor().cuda()"
- replace these with corresponding CPU tensor "torch.FloatTensor(idx_tensor)
- Change device for model evaluation and data loading to CPU: "model.to("cpu")", "img.to("cpu")"
- Apply changes also to the detector the model uses.
  detector.py -> class RetinaFace: this class was built for GPU. It has been modified to make it work for CPU. Set gpu_id = -1.
