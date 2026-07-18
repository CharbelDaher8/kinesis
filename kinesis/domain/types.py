"""Value objects that flow between stages: Frame, HandObservation, HandFeatures, GestureEvent, Intent. Keep them immutable — they are the stable contracts."""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    frame_id: int

@dataclass
class HandObservation:
    landmarks: np.ndarray
    #handedness flipped form open cv by default so we take care of this
    handedness: str
    timestamp: float


@dataclass
class HandFeatures:
    cursor_point: tuple          # (x, y) index-tip position, normalized 0..1
    pinch_distance: float        # thumb-index gap / hand scale (scale-invariant)
    fingers_extended: tuple      # (thumb, index, middle, ring, pinky) bools
    centroid: tuple              # (x, y) hand center, normalized
    velocity: tuple              # (vx, vy) centroid motion per second
    palm_normal: tuple           # (nx, ny, nz) unit normal of the palm plane
    palm_angle: float            # in-plane hand rotation, radians
    timestamp: float

class GestureType(Enum):
    POINT = "point"
    PINCH = "pinch"
    SWIPE = "swipe"


class GesturePhase(Enum):
    BEGAN = "began"
    CHANGED = "changed"
    ENDED = "ended"
    RECOGNIZED = "recognized"


@dataclass
class GestureEvent:
    type: GestureType
    phase: GesturePhase
    hand: str  
    timestamp: float
    data: dict = field(default_factory=dict)


class IntentType(Enum):
    """Semantic actions the output adapters know how to perform."""
    MOVE_CURSOR = "move_cursor"
    CLICK = "click"
    SCROLL = "scroll"
    ZOOM = "zoom"
    NEXT_PAGE = "next_page"
    PREV_PAGE = "prev_page"


@dataclass
class Intent:
    type: IntentType
    timestamp: float
    # Arguments the action needs, kept resolution-independent so the same intent
    # works on any monitor (the adapter maps normalized coords to real pixels):
    #   MOVE_CURSOR -> {"x": 0..1, "y": 0..1}
    #   SCROLL      -> {"dx": ..., "dy": ...}
    #   ZOOM        -> {"factor": ...}
    #   CLICK       -> {"button": "left"}
    params: dict = field(default_factory=dict)

