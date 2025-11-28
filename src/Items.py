import random

class Items:
    itemType, isRepeating, itemAmount = 0, False, 0

    def __init__(self, type: str, repeat: bool, amount: int):
        self.itemType = type
        self.isRepeating = repeat
        self.itemAmount = amount 

    def getAmount(self):
        if self.isRepeating:
            return self.itemAmount
        else:
            temp = self.itemAmount
            self.itemAmount = 0
            return temp
        
    def getType(self):
        return self.itemType