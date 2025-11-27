import Terrain
import Tiles
import Items
import Trader
import Player

# TODO: adjust as needed, just example for now
ITEM_PROBABILITY_TABLE = [0.5, 0.1, 0.01]


class Map:
    width = 0
    height = 0
    player = None
    difficulty = -1
    tile_map = None

    def reset(self, width=0, height=0, player=None, difficulty=-1):
        regenerate = False
        if (width > 0):
            regenerate = True
            self.width = width
        if (height > 0):
            regenerate = True
            self.height = height
        if (not(player is None)):
            self.player = player
        if (difficulty >= 0):
            regenerate = True
            self.difficulty = difficulty
        if (regenerate):
            self.tile_map = []
            for x in range(0, width):
                current_row = []
                for y in range(0, height):
                    # TODO: implement tile generation
                    current_tile = Tile()
                    current_row.append(current_tile)
                self.tile_map.append(current_row)

    def __init__(self, width, height, player, difficulty):
        self.reset(width, height, player, difficulty)
