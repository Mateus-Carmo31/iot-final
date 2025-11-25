# Split Inference System (YOLOv8)

> Project members:

> Heitor Henrique Da Silva

> Mateus Carmo de Oliveira

> Milena de Faria Silva

This project runs a split YOLOv8 model. The **Client** (TV Box) runs the initial layers ("head" model) and sends data to the **Server** which finishes processing ("tail" model) and counts objects.

## Setup (Both Machines)

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

1.  **Create environment and install dependencies:**

    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv sync
    ```

2.  **Required File Structure:**
    Ensure your directory looks like this on both machines (or minimal required files per side):

    ```text
    ├── models/               # Contains /head/ and /tail/ weights
    ├── park_ic2/             # (Client only) Folder containing parking images
    ├── yolov8.yaml           # Model config
    ├── mask_original_...png  # (Server only) Mask image
    ├── server.py
    ├── client.py
    ├── yolo8_modules.py
    └── split_schema_pb2...   # Generated gRPC files
    ```
    
    The client only requires the models in the `head/` folder, and the same is the case for the server and the `tail/` folder. The unused folders can be deleted in case of resource constraints.

-----

## Server

Starts the gRPC server to listen for incoming tensors, run the model tail, and log results.

```bash
# Usage: --model_size [n,s,m,l,x] --split [a,b,c]
uv run server.py --model_size n --split a
```

  * **Port:** Opens on `0.0.0.0:8000`.

-----

## Client (TV Box)

Process images locally and offload the rest to the server.

```bash
# Usage: Point --server to your strong machine's IP
uv run client.py --server <SERVER_IP>:8000 --model_size n --split a --image_count 10
```

### Common Arguments

  * `--server`: IP and port of the server (e.g., `192.168.1.50:8000`).
  * `--model_size`: Must match the server (default: `n`).
  * `--split`: Split point configuration (must match server).
  * `--shuffle`: Randomize image order.
