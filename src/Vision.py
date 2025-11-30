#import Map
#import Player

class Vision:
    player_reference = None
    vision_radius = 4
    closeList = [(0, 0)]


    def setVisionRadius(self, vision_radius):
        self.vision_radius = vision_radius
        self.assembleCloseList()

    def assembleCloseList(self):
        self.closeList = []
        for i in range(0, self.vision_radius + 1):
            for k in sorted(range(i * -1, i + 1), key=abs):
                self.closeList.append((i - abs(k), k))

    def inBounds(self, pos, size):
        return (
            (pos[0] >= 0 and pos[1] >= 0) and
            (pos[0] < size[0] and pos[1] < size[1])
        )

    def closestItem(self, itemType):
        map_reference = self.player_reference.map_reference
        size = (map_reference.width, map_reference.width)
        for rel_pos in self.closeList:
            true_pos = (self.player_reference.position[0] + rel_pos[0], self.player_reference.position[1] + rel_pos[1])
            if self.inBounds(true_pos, size):
                current_tile = map_reference.getTile(true_pos[0], true_pos[1])
                if current_tile.getItem.getType == itemType:
                    return true_pos
        return None

    def secondClosestItem(self, itemType):
        map_reference = self.player_reference.map_reference
        size = (map_reference.width, map_reference.width)
        found_first = False
        for rel_pos in self.closeList:
            true_pos = (self.player_reference.position[0] + rel_pos[0], self.player_reference.position[1] + rel_pos[1])
            if self.inBounds(true_pos, size):
                current_tile = map_reference.getTile(true_pos[0], true_pos[1])
                if current_tile.getItem.getType == itemType:
                    if not found_first:
                        found_first = True
                    else:
                        return true_pos
        return None


    def attach_player(self, player):
        self.player_reference = player


myVis = Vision()
myVis.assembleCloseList()
print(myVis.closeList)
