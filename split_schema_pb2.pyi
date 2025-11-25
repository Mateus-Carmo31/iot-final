from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Tensor(_message.Message):
    __slots__ = ("is_null", "raw_data", "shape", "dtype")
    IS_NULL_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    is_null: bool
    raw_data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    def __init__(self, is_null: bool = ..., raw_data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ...) -> None: ...

class ClientRequest(_message.Message):
    __slots__ = ("client_name", "layers_output", "img_name", "img_shape", "orig_shape", "last_layer_idx", "timestamp", "inference_time", "message_send_timestamp", "client_ram_mb", "client_cpu_perc")
    CLIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    LAYERS_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    IMG_NAME_FIELD_NUMBER: _ClassVar[int]
    IMG_SHAPE_FIELD_NUMBER: _ClassVar[int]
    ORIG_SHAPE_FIELD_NUMBER: _ClassVar[int]
    LAST_LAYER_IDX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_TIME_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_SEND_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CLIENT_RAM_MB_FIELD_NUMBER: _ClassVar[int]
    CLIENT_CPU_PERC_FIELD_NUMBER: _ClassVar[int]
    client_name: str
    layers_output: _containers.RepeatedCompositeFieldContainer[Tensor]
    img_name: str
    img_shape: _containers.RepeatedScalarFieldContainer[int]
    orig_shape: _containers.RepeatedScalarFieldContainer[int]
    last_layer_idx: int
    timestamp: float
    inference_time: float
    message_send_timestamp: float
    client_ram_mb: float
    client_cpu_perc: float
    def __init__(self, client_name: _Optional[str] = ..., layers_output: _Optional[_Iterable[_Union[Tensor, _Mapping]]] = ..., img_name: _Optional[str] = ..., img_shape: _Optional[_Iterable[int]] = ..., orig_shape: _Optional[_Iterable[int]] = ..., last_layer_idx: _Optional[int] = ..., timestamp: _Optional[float] = ..., inference_time: _Optional[float] = ..., message_send_timestamp: _Optional[float] = ..., client_ram_mb: _Optional[float] = ..., client_cpu_perc: _Optional[float] = ...) -> None: ...

class ServerReply(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...
