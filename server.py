from concurrent import futures
import csv
import os
import time
from fastapi import Request
import torch
from torch.nn import Module
from ultralytics.engine.results import Results
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel
from PIL import Image
import argparse
import grpc
import split_schema_pb2
import split_schema_pb2_grpc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_file = 'mask_original_img_768_1024_bw.png'
mask = Image.open(mask_file)
server_args = None

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--model_size', default='n', type=str, choices=['n', 's', 'm', 'l', 'x'], help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--split', default='a', type=str, choices=['a', 'b', 'c'], help='Split configuration to use (a, b, c)')
    parser.add_argument('--log_name', default='teste', type=str, help='File where results will be logged')

    return parser

parser = argparse.ArgumentParser('', parents=[get_args_parser()])
args = parser.parse_args()
METRICS_FILE = f'./results/metrics_{args.model_size}_{args.split}.csv'

def xywh_to_xyxy(boxes):
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def nms(boxes, scores, iou_threshold=0.5, max_det=300):
    if len(boxes) == 0:
        return []
    
    boxes = xywh_to_xyxy(boxes)  # Ensure correct format

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by confidence score

    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep

def detection_matrix_modified(x,y,mask):
    mask = mask[:,:,0]
    # print(mask.shape)
    mask_h, mask_w = mask.shape
    # mask_x,mask_y = mask.shape[:2]
    # mask_x, mask_y = (480,640)
    x = x*mask_w
    y = y*mask_h
    x_idx = int(np.clip(x, 0, mask_w - 1))
    y_idx = int(np.clip(y, 0, mask_h - 1))
    pixel_value = mask[int(y_idx),int(x_idx)]
    # print(f"\n points are {x},{y} \n pixel value: {pixel_value} and mask shape is {mask.shape}\n mask_x = {mask_x}, mask_y = {mask_y}")
    if pixel_value == 255:
        # print("The point is outside the mask.")
        return False
    else:
        # print("The point is inside the mask.")
        return True

def count_cars_post(lines, mask,class_names_dict=0):

    car_count = 0
    truck_count = 0

    for line in lines:

        line_ = np.array(line)
        x_center, y_center, width, height = line_

        point_inside = detection_matrix_modified(x_center,y_center,mask)
        # print(f"\n\n\n\n point {x_center}, {y_center} is {point_inside} ")
        if point_inside == True:
            
            car_count += 1

    return car_count + truck_count

def count_cars(results,mask, th=0.25):
    """
    Runs prediction and counts cars using an Ultralytics model.
    """

    detected_boxes = []
    
    # Get the bounding boxes (xywh format) and scores
    boxes_xywh = results.boxes.xywhn.cpu().numpy() # Normalized (cx, cy, w, h)
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for i in range(len(scores)):
        # Filter by confidence and class (2=car, 7=truck)
        if scores[i] >= th and classes[i] in [2, 7]:
            detected_boxes.append(boxes_xywh[i])

    if detected_boxes:
        detected_boxes = np.array(detected_boxes)
        # We need the scores for NMS, not the box height!
        detected_scores = scores[np.where(np.isin(classes, [2, 7]) & (scores >= th))]

        # Run NMS
        keep = nms(detected_boxes, detected_scores, iou_threshold=0.45)
        final_boxes = detected_boxes[keep]

        # Count cars inside the mask
        cars = count_cars_post(final_boxes, mask)
        return cars, final_boxes
    else:
        return 0, []

def init_metrics_file(log_name):
    log_file = log_name if log_name else METRICS_FILE
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Add new headers
        writer.writerow([
            'timestamp',
            'client_name',
            'image_name',
            'client_inference_ms',
            'client_comm_ms',
            'server_inference_ms',
            'total_time_ms',
            'model_size',
            'split_layer',
            'payload_size_mib',
            'client_cpu_perc',
            'client_ram_mb',
            'cars_found'
        ])

def log_metrics(
        client_name,
        image_name,
        client_start_time,
        client_inference_time,
        client_comm_time,
        server_inference_time,
        client_cpu,
        client_ram,
        payload_size,
        car_count,
        log_name
):
    global args

    server_time = time.time()
    # Calculate total end-to-end time from client start to server finish
    total_time_ms = (server_time - client_start_time) * 1000

    log_file = log_name if log_name else METRICS_FILE
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{server_time * 1000:.2f}",
            client_name,
            image_name,
            f"{client_inference_time * 1000:.2f}",
            f"{client_comm_time * 1000:.2f}",
            f"{server_inference_time * 1000:.2f}",
            f"{total_time_ms:.2f}",
            args.model_size,
            args.split,
            f"{payload_size:.2f}",
            f"{client_cpu:.2f}",
            f"{client_ram:.2f}",
            car_count
        ])

def load_tail_model():
    """Loads the tail model into memory."""
    global args
    
    default_splits = {
        "a": (10, [4, 6, 9]),
        "b": (16, [9, 12, 15]),
        "c": (22, [15, 18, 21])
    }
    
    config_path = './yolov8.yaml'
    
    splits = default_splits.get(args.split, default_splits["a"])
    split_layer, _ = splits
    
    model_size = args.model_size
    
    print("Building tail model structure from YAML...")
    model_tail = SplitDetectionModel(cfg=config_path, split_layer=split_layer,scale=model_size)
    
    print(f"Loading tail weights from tail_weights.pt...")
    model_tail.tail.load_state_dict(torch.load(f'./models/tail/{model_size}/tail_weights_{args.split}.pt', map_location=device))
    model_tail.eval()
    model_tail.to(device)
    print("Tail model loaded and ready on", device)
    return model_tail

def predict_tail(model: Module, y_head: dict, image_size: int = 640) -> Results:
    predictor = SplitDetectionPredictor(model, overrides={"imgsz": image_size})
    predictions = model.forward_tail(y_head)
    return predictor.postprocess(predictions, y_head["img_shape"], y_head["orig_shape"])[0]

class SplitInferenceServicer(split_schema_pb2_grpc.SplitInferenceServiceServicer):
    def __init__(self, args, model_tail):
        self.args = args
        self.model_tail = model_tail

    def proto_tensor_to_torch(self, proto_tensor):
        if proto_tensor.is_null:
            return None
        # Convert raw bytes back to numpy, then to torch on correct device
        np_array = np.frombuffer(proto_tensor.raw_data, dtype=proto_tensor.dtype).reshape(tuple(proto_tensor.shape))
        return torch.as_tensor(np_array, device=device)

    def SendActivations(self, request, context):
        received_time = time.time()
        payload_size_mb = request.ByteSize() / (1024 * 1024)

        try:
            # 1. Deserialize Tensors
            y_head = {
                "layers_output": [self.proto_tensor_to_torch(t) for t in request.layers_output],
                "img_shape": tuple(request.img_shape),
                "orig_shape": tuple(request.orig_shape),
                "last_layer_idx": request.last_layer_idx,
            }

            # 2. Run Tail Inference
            print(f"Running inference for {request.client_name} - {request.img_name}")
            start_inference = time.time()
            with torch.no_grad():
                results = predict_tail(self.model_tail, y_head)
            server_inference_time = time.time() - start_inference

            results.orig_shape = y_head["orig_shape"]
    
            if results.boxes:
                results.boxes.orig_shape = y_head["orig_shape"]
            
            # 3. Post-process and Count
            h, w = y_head["orig_shape"]
            mask_resized = np.array(mask.resize((w, h), Image.NEAREST))
            cars, _ = count_cars(results, mask_resized)

            # 4. Log Metrics
            client_comm_time = received_time - request.message_send_timestamp
            log_metrics(request.client_name, request.img_name, request.timestamp, 
                        request.inference_time, client_comm_time, server_inference_time, 
                        request.client_cpu_perc, request.client_ram_mb, payload_size_mb, 
                        cars, self.args.log_name)

            print(f"Complete. Cars detected: {cars}")
            return split_schema_pb2.ServerReply(success=True, message=f"Processed, cars: {cars}")

        except Exception as e:
            print(f"Error processing request: {e}")
            return split_schema_pb2.ServerReply(success=False, message=str(e))

def serve():
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    init_metrics_file(args.log_name)
    model_tail = load_tail_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_receive_message_length', 100 * 1024 * 1024)]) # 100MB max
    split_schema_pb2_grpc.add_SplitInferenceServiceServicer_to_server(SplitInferenceServicer(args=args,model_tail=model_tail), server)
    
    server.add_insecure_port(f'[::]:8000')
    print(f"gRPC Server starting ...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
