
import argparse
from typing import Union, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel
from ultralytics.nn.tasks import DetectionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_head(model: Module, input_image: Union[str, Tensor], save_layers: List[int], image_size: int = 640) -> dict:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})

    # Prepare data
    predictor.setup_source(input_image)
    batch = next(iter(predictor.dataset))
    predictor.batch = batch
    path, input_image, _, _ = batch

    # Preprocess
    preprocessed_image = predictor.preprocess(input_image)
    if isinstance(input_image, list):
        input_image = np.array([np.moveaxis(img, -1, 0) for img in input_image])

    # Head predict
    y_head = model.forward_head(preprocessed_image, save_layers)
    y_head["img_shape"] = preprocessed_image.shape[2:]
    y_head["orig_shape"] = input_image.shape[2:]

    return y_head


def predict_tail(model: Module, y_head: dict, image_size: int = 640) -> Results:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})

    # Tail predict
    predictions = model.forward_tail(y_head)

    # Postprocess
    yolo_results = predictor.postprocess(predictions, y_head["img_shape"], y_head["orig_shape"])[0]

    return yolo_results


if __name__ == "__main__":
    config_path = './yolov8.yaml'
    
    default_splits = (10, [4, 6, 9])
    
    splits = default_splits
    split_layer, save_layers = default_splits
    
    image_path = r"./original_1653085977858.jpg" # 👈 Use a raw string (r"...") for Windows paths
    
    print("Building model structure from YAML...")
    # empty_pretrained_model = DetectionModel(cfg=config_path, ch=3)
    
    model_head = SplitDetectionModel(cfg=config_path, split_layer=split_layer)
    
    print(f"Loading head weights from head_weights.pt...")
    model_head.head.load_state_dict(torch.load('head_weights.pt', map_location=torch.device('cpu')))
    model_head.eval()
    model_head.to(device)
    print("Head model loaded successfully.")
    
    head_output = predict_head(model_head, image_path, save_layers)
    
    
    model_tail = SplitDetectionModel(cfg=config_path, split_layer=split_layer)
    model_tail.tail.load_state_dict(torch.load('tail_weights.pt', map_location=torch.device('cpu')))
    model_tail = model_tail.eval()
    model_tail = model_tail.to(device)
    results = predict_tail(model_tail, head_output)
    
    original_image = cv2.imread(image_path)
    
    annotated_image = results.plot(img=original_image)
    
    # Save the annotated image to a file
    cv2.imwrite("output_prediction.jpg", annotated_image)
    
    print("Successfully saved prediction to 'output_prediction.jpg'")
    print(f"Boxes found: {results.boxes.xywh}")