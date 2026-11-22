"""ASVspoof protocol parsing and waveform datasets."""

from .protocol import LABEL_TO_TARGET, ProtocolEntry, load_protocol

__all__ = ["LABEL_TO_TARGET", "ProtocolEntry", "load_protocol"]
