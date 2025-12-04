"""Top-level package for the Wilderness Survival System (WSS).

Convenience re-exports are provided for commonly used modules and classes so
that examples and training code can import from ``src`` directly.

Example
-------
>>> from src import Map, Player, Terrain
>>> p = Player()
>>> m = Map(20, 10, p, difficulty="medium")
"""

# Make src a package and re-export common modules
from . import Terrain
from .Map import Map
from .Player import Player
from .Tiles import Tile
from . import Items
from . import Trader
from . import Difficulty
