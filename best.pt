from cog import BasePredictor, Input, Path
from ultralytics import YOLO
import cv2
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        self.model = YOLO("best.pt")  # Replace with your actual model filename

    def predict(
        self,
        image: Path = Input(description="Input image")
    ) -> Path:
        results = self.model(str(image))
        result = results[0]

        if not result.masks:
            raise Exception("No mask found")

        mask = result.masks.data[0].cpu().numpy() * 255
        mask = cv2.resize(mask, (result.orig_shape[1], result.orig_shape[0]))
        mask_path = "/tmp/output.png"
        cv2.imwrite(mask_path, mask)
        return Path(mask_path)
