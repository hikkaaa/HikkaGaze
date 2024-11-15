import argparse
import os
import numpy as np
import cv2
import math
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from utils import select_device, draw_gaze, classify_distraction
from face_detection import RetinaFace
from model import L2CS


class GazeDemo:
    def __init__(self, snapshot_path=None, cam_id=0, arch='ResNet50', threshold=120, uninterested_threshold=10):
        """
        Initialize the GazeDemo class.
        Args:
            snapshot_path (str): Path to the pre-trained model.
            cam_id (int): Camera device ID to use for capturing video.
            arch (str): Network architecture (e.g., ResNet50).
            threshold (int): Distance threshold to classify distraction.
            uninterested_threshold (int): Count threshold to classify the user as "UNINTERESTED."
        """
        self.snapshot_path = snapshot_path or './models/L2CSNet_gaze360.pkl'
        self.cam_id = cam_id
        self.arch = arch
        self.threshold = threshold
        self.uninterested_threshold = uninterested_threshold
        self.device = select_device()
        self.model = self._load_model()
        self.detector = RetinaFace(gpu_id=-1)
        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)
        self.transformations = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.distracted_count = 0  # Counter for consecutive distracted frames

    def _load_model(self):
        """
        Load the pre-trained model.
        """
        print("Loading model...")
        model = self.get_architecture(self.arch, 90)
        saved_state_dict = torch.load(self.snapshot_path, map_location=self.device)
        model.load_state_dict(saved_state_dict)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def get_architecture(arch, bins):
        """
        Get the network architecture.
        """
        if arch == 'ResNet18':
            return L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        elif arch == 'ResNet34':
            return L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        elif arch == 'ResNet101':
            return L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        elif arch == 'ResNet152':
            return L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
        else:
            return L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)

    def run(self):
        """
        Run the gaze demo using the webcam.
        """
        print("Starting webcam...")
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        with torch.no_grad():
            while True:
                success, frame = cap.read()
                if not success:
                    break

                faces = self.detector(frame)
                if faces is not None:
                    for box, landmarks, score in faces:
                        if score < 0.95:
                            continue

                        x_min, y_min, x_max, y_max = map(int, box)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        # Crop and preprocess the image
                        img = frame[y_min:y_max, x_min:x_max]
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = self.transformations(Image.fromarray(img))
                        img = img.unsqueeze(0).to(self.device)

                        # Predict gaze
                        gaze_pitch, gaze_yaw = self.model(img)
                        pitch_predicted = torch.sum(self.softmax(gaze_pitch).data[0] * self.idx_tensor) * 4 - 180
                        yaw_predicted = torch.sum(self.softmax(gaze_yaw).data[0] * self.idx_tensor) * 4 - 180
                        pitch_predicted = pitch_predicted.cpu().numpy() * np.pi / 180.0
                        yaw_predicted = yaw_predicted.cpu().numpy() * np.pi / 180.0

                        # Draw gaze and classify distraction
                        frame, origin, tip = draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                                                       (pitch_predicted, yaw_predicted))
                        is_distracted = classify_distraction(origin, tip, threshold=self.threshold)

                        if is_distracted:
                            self.distracted_count += 1
                        else:
                            self.distracted_count = 0

                        # Display distraction status
                        if self.distracted_count > self.uninterested_threshold:
                            cv2.putText(frame, "UNINTERESTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        elif is_distracted:
                            cv2.putText(frame, "Distracted!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                cv2.imshow("Gaze Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Initialize and run the demo
    demo = GazeDemo()
    demo.run()
