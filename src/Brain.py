class Brain:
    player_reference = None
    vision_reference = None

    def attach_player(self, player):
        self.player_reference = player

    def attach_vision(self, vision):
        self.vision_reference = vision
