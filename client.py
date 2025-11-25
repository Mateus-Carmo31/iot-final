import json
import os
import requests
import time
from typing import Union, List
import argparse
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm
from yolo8_modules import SplitDetectionPredictor, SplitDetectionModel
import psutil
import threading
import grpc
import split_schema_pb2
import split_schema_pb2_grpc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = './park_ic2'

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--model_size', default='n', type=str, choices=['n', 's', 'm', 'l', 'x'], help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--split', default='a', type=str, choices=['a', 'b', 'c'], help='Split configuration to use (a, b, c)')
    parser.add_argument('--server', default='localhost:8000', type=str, help='address + port of the server endpoint')
    parser.add_argument('--client_name', default='client', type=str, help='client name for logging by the server')
    parser.add_argument('--image_count', default=1, type=int, help='how many images to run inference on')
    parser.add_argument('--shuffle', default=False, action=argparse.BooleanOptionalAction,
                        help='shuffle the order of images in the dataset before running inference on them')

    return parser

def monitor_process(sample_interval=0.1):
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    while True:
        cpu = process.cpu_percent(interval=sample_interval) / psutil.cpu_count(logical=True)
        mem = process.memory_info().rss
        yield cpu, mem
        time.sleep(sample_interval)

cpu_readings = []
mem_readings = []
stop = False

def sampler():
    for cpu, mem in monitor_process(0.01):
        if stop: break
        cpu_readings.append(cpu)
        mem_readings.append(mem)

def get_process_info():
    avg_cpu = sum(cpu_readings) / len(cpu_readings)
    avg_mem = sum(mem_readings) / len(mem_readings)
    # print(cpu_readings)
    return (avg_cpu, avg_mem)

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
    with torch.no_grad():
        y_head = model.forward_head(preprocessed_image, save_layers)

    y_head["img_shape"] = preprocessed_image.shape[2:]
    y_head["orig_shape"] = input_image.shape[2:]

    return y_head

def create_partial_model(cfg,split_layer,split_layer_letter,model_size):
    model_head = SplitDetectionModel(cfg=cfg, split_layer=split_layer, scale=model_size)
    model_head.head.load_state_dict(torch.load(f'./models/head/{model_size}/head_weights_{split_layer_letter}.pt', map_location=torch.device('cpu')))
    model_head.eval()
    model_head.to(device)
    return model_head

def torch_to_proto_tensor(tensor_or_none: Union[torch.Tensor, None]) -> split_schema_pb2.Tensor:
    """Converts a PyTorch tensor to our defined Proto Tensor message efficiently."""
    if tensor_or_none is None:
        return split_schema_pb2.Tensor(is_null=True)
    
    # Ensure it's on CPU and numpy format for easy serialization
    np_array = tensor_or_none.detach().cpu().numpy()
    
    return split_schema_pb2.Tensor(
        is_null=False,
        raw_data=np_array.tobytes(),
        shape=np_array.shape,
        dtype=str(np_array.dtype)
    )

def send_activations_to_server(stub,client_name,image_name,head_output,start_time,inference_time,client_cpu_perc,client_ram_mb):
    message_send_timestamp = time.time()
    
    proto_layers = [torch_to_proto_tensor(t) for t in head_output["layers_output"]]
    
    request = split_schema_pb2.ClientRequest(
        client_name=client_name,
        layers_output=proto_layers,
        img_name=image_name,
        img_shape=list(head_output["img_shape"]),
        orig_shape=list(head_output["orig_shape"]),
        last_layer_idx=int(head_output["last_layer_idx"]),
        timestamp=start_time,
        inference_time=inference_time,
        message_send_timestamp=message_send_timestamp,
        client_ram_mb=client_ram_mb / 1024 / 1024,
        client_cpu_perc=client_cpu_perc
    )
    
    try:
        # You might want to increase timeout for large tensor transfers if needed
        response = stub.SendActivations(request, timeout=30)
        if not response.success:
             print(f"Server reported error: {response.message}")
             
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    
    default_splits = {
        "a": (10, [4, 6, 9]),
        "b": (16, [9, 12, 15]),
        "c": (22, [15, 18, 21])
    }
    
    splits = default_splits.get(args.split, default_splits["a"])
    split_layer, save_layers = splits

    model_size = args.model_size

    partial_model    = create_partial_model(cfg='./yolov8.yaml',split_layer=split_layer,split_layer_letter=args.split,model_size=model_size)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

    # Filter images by image_count, and sample randomly from the full list if
    # shuffle is on
    if args.image_count > 0:
        if args.shuffle:
            image_files = np.random.choice(image_files, size=args.image_count, replace=False)
        else:
            image_files = image_files[:args.image_count]

    MAX_MESSAGE_LENGTH = 100 * 1024 * 1024 # 100MB
    options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]
    
    channel = grpc.insecure_channel(args.server, options=options)
    stub = split_schema_pb2_grpc.SplitInferenceServiceStub(channel)

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(IMAGE_DIR, image_file)

        stop = False
        t = threading.Thread(target=sampler)
        t.start()
        start_time = time.time()
        head_output = predict_head(partial_model, image_path, save_layers) # Inference
        inference_time = time.time() - start_time
        stop = True
        t.join()
        avg_cpu, avg_mem = get_process_info()
        cpu_readings.clear()
        mem_readings.clear()
        send_activations_to_server(stub,args.client_name,image_file,head_output,start_time,inference_time,avg_cpu,avg_mem)
        # end_time = time.time()
