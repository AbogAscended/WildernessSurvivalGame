import Items, Trader

class Player:
    maxFood = maxWater = maxStrength = 0
    currentGold = currentWater = currentFood = 0
    position = (0, 0)
    map_reference = None

    def __init__(self):
        self.reset()
    
    def reset(self):
        # global maxWater, maxFood, maxStrength, currentGold, currentWater, currentFood
        self.maxFood = self.maxWater = self.maxStrength = 100
        self.currentGold = 20
        self.currentWater = 10
        self.currentFood = 10
        self.position = (0, 0)

    def attach_map(self, new_map):
        map_reference = new_map
    
    def bargain(trader): # returns 1 for sucessful trade, 0 for unsucessful
        bargaining = True
        while bargaining:
            proposal = trader.getProposal()
            print("trader wants:")
            print(proposal[0])
            print("for:")
            print(proposal[1])

            user_input = input("do you accept the offer?: y/n \n")
            if user_input == "y":
                print("trade accepted")
                Player.transfer(Player, proposal)
                return 1 
            elif user_input == "n":
                user_input = input("do you want to propose a new offer?: y/n \n")
                if user_input == "n":
                    return 0 # transfer will not intiate
                elif user_input == "y": # bartering begins
                    user_input = input("Enter what you want in values in order of gold, water, food separated by spaces \n")
                    recieving = [int(x) for x in user_input.split()]
                    user_input = input("Enter what you will trade away in values in order of gold, water, food separated by spaces \n")
                    offering = [int(x) for x in user_input.split()]
                    
                    result = trader.recieveProposal([offering, recieving])
                    if result == 0: # trader accepts new offer
                        print("trade accepted")
                        Player.transfer(Player, [offering, recieving]) # intiate transfer
                        return 1
                    elif result == 1: # trader rejects new offer
                        print("trader has rejected your offer, this is their new offer:")
                    elif result == 2: # trader stops all trading
                        print("trader is no longer willing to trade. trading session has ended")
                        return 0

    def collect(self, item: Items): # only for collecting items
        type, amount = item.getType(), item.getAmount()
        match type:
            case "water":
                self.currentWater += amount
            case "food":
                self.currentFood += amount
            case "gold":
                self.currentGold += amount

    def transfer(self, offer): # only for bargaining with trader
        # global currentGold, currentWater, currentFood
        self.currentGold += (offer[1][0] - offer[0][0])
        self.currentWater += (offer[1][1] - offer[0][1])
        self.currentFood += (offer[1][2] - offer[0][2])

    def getResources(self):
        return "gold:" + str(self.currentGold) + " water: " + str(self.currentWater) + " food: " + str(self.currentFood)
