from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np

from ._sockutil import Socket
from .affetto import Affetto

if TYPE_CHECKING:
    import socket
    from collections.abc import Callable
    from pathlib import Path


R = TypeVar("R")


def split_received_msg(
    data: bytes | str,
    function: Callable[[str], R] = float,  # type: ignore[assignment]
    sep: str | None = None,
    *,
    strip: bool = True,
) -> list[R]:
    """Return a list of values converted from received bytes."""
    if isinstance(data, bytes):
        decoded_data = data.decode()
    elif isinstance(data, str):
        decoded_data = data
    else:
        msg = f"unsupported type: {type(data)}"
        raise TypeError(msg)
    if strip:
        decoded_data = decoded_data.strip(sep)
    return list(map(function, decoded_data.split(sep)))


def convert_array_to_string(
    array: list[float] | list[int] | np.ndarray,
    sep: str = " ",
    f_spec: str = ".0f",
    precision: int | None = None,
) -> str:
    """Return a string of array joined with specific format."""
    if precision is None:
        formatted_array = [f"{x:{f_spec}}" for x in array]
    else:
        formatted_array = [f"{x:.{precision}f}" for x in array]
    return sep.join(formatted_array)


def convert_array_to_bytes(
    array: list[float] | list[int] | np.ndarray,
    sep: str = " ",
    f_spec: str = ".0f",
    precision: int | None = None,
) -> bytes:
    """Return bytes encoded array joined with specific format."""
    return convert_array_to_string(array, sep, f_spec, precision).encode()


def unzip_array_as_ndarray(array: list[float] | list[int] | np.ndarray, ncol: int = 3) -> np.ndarray:
    ret = np.array(array).reshape((int(len(array) / ncol), ncol))
    return ret.T


def unzip_array(array: list[float] | list[int] | np.ndarray, n: int = 3) -> list[Any]:
    arr = unzip_array_as_ndarray(array, ncol=n)
    return arr.tolist()


def zip_arrays_as_ndarray(
    *arrays: list[float] | list[int] | np.ndarray,
) -> np.ndarray:
    arr = np.stack(arrays, axis=1)
    return arr.flatten()


@overload
def zip_arrays(*arrays: list[float]) -> list[float]: ...


@overload
def zip_arrays(*arrays: list[int]) -> list[int]: ...


@overload
def zip_arrays(*arrays: np.ndarray) -> list[float]: ...


def zip_arrays(
    *arrays: list[float] | list[int] | np.ndarray,
) -> list[float] | list[int]:
    return list(zip_arrays_as_ndarray(*arrays))


class AffComm(Affetto):
    comm_config: dict[str, Any]
    sensory_socket: Socket
    command_socket: Socket

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.sensory_socket = Socket()
        self.command_socket = Socket()
        super().__init__(config_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        try:
            cpath = self._config_path
        except AttributeError:
            cpath = None
        return f"""\
AffComm configuration:
  Config file: {cpath!s}
   Receive at: {self.sensory_socket!s}
      Send to: {self.command_socket!s}
"""

    def load_config(self, config_dict: dict[str, Any]) -> None:
        super().load_config(config_dict)
        self.load_comm_config()

    def load_comm_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.comm_config = c["comm"]
        self.sensory_socket.addr = self.comm_config["local"]
        self.command_socket.addr = self.comm_config["remote"]

    def create_sensory_socket(self, addr: tuple[str, int] | None = None) -> socket.socket:
        s = self.sensory_socket.create()
        self.sensory_socket.bind(addr)
        return s

    def create_command_socket(self) -> socket.socket:
        return self.command_socket.create()

    def create_sockets(self, sensory_addr: tuple[str, int] | None = None) -> tuple[socket.socket, socket.socket]:
        return (self.create_sensory_socket(sensory_addr), self.create_command_socket())

    def close_sensory_socket(self) -> None:
        if self.sensory_socket.is_created():
            self.sensory_socket.close()

    def close_command_socket(self) -> None:
        if self.command_socket.is_created():
            self.command_socket.close()

    def close(self) -> None:
        self.close_sensory_socket()
        self.close_command_socket()

    def receive(self, bufsize: int = 1024) -> bytes:
        return self.sensory_socket.recvfrom(bufsize)

    def receive_as_list(
        self,
        bufsize: int = 1024,
        function: Callable[[str], R] = float,  # type: ignore[assignment]
    ) -> list[R]:
        return split_received_msg(self.sensory_socket.recvfrom(bufsize), function=function)

    def receive_as_2darray(
        self,
        bufsize: int = 1024,
    ) -> np.ndarray:
        sarr = split_received_msg(self.sensory_socket.recvfrom(bufsize), function=float)
        return unzip_array_as_ndarray(sarr, ncol=3)

    def send(self, send_bytes: bytes, addr: tuple[str, int] | None = None) -> int:
        return self.command_socket.sendto(send_bytes, addr)

    def send_commands(
        self,
        *arrays: np.ndarray,
        addr: tuple[str, int] | None = None,
    ) -> int:
        return self.command_socket.sendto(convert_array_to_bytes(zip_arrays_as_ndarray(*arrays)), addr)
