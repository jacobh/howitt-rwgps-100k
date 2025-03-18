from typing import Any, BinaryIO

def pack(o: Any, stream: BinaryIO, **kwargs: Any) -> None: 
    """
    Pack object `o` and write it to `stream`.

    See :class:`Packer` for options.
    """
    ...

def packb(o: Any, **kwargs: Any) -> bytes:
    """
    Pack object `o` and return packed bytes.

    See :class:`Packer` for options.
    """
    ...

def unpack(stream: BinaryIO, **kwargs: Any) -> Any:
    """
    Unpack an object from `stream`.

    Raises `ExtraData` when `stream` contains extra bytes.
    See :class:`Unpacker` for options.
    """
    ...

def unpackb(packed: bytes, **kwargs: Any) -> Any:
    """
    Unpack an object from packed bytes.

    See :class:`Unpacker` for options.
    """
    ...

# Aliases for compatibility with simplejson/marshal/pickle.
load = unpack
loads = unpackb
dump = pack
dumps = packb