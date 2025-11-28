import random, Items

class Trader(Items.Items):
    deviation = maxCounters = totalCounters = proposal = 0

    def __init__(self, difficulty):
        super().__init__("trader", True, float("inf"))
        # global deviation, maxCounters, proposal, totalCounters
        rand_float = random.random()

        match difficulty: # difficulty scaling for trader: extreme, hard, normal, easy
            case "extreme":
                if rand_float >= 0.2: #aggressive
                    self.deviation = 0.5
                elif rand_float >= 0.05: #assertive
                    self.deviation = 1.0
                else: #passive
                    self.deviation = 1.5
            case "hard":
                if rand_float >= 0.66: #aggressive
                    deviation = 0.5
                elif rand_float >= 0.33: #assertive
                    self.deviation = 1.0
                else:
                    self.deviation = 1.5
            case "normal":
                if rand_float >= 0.4: #passive
                    self.deviation = 1.5
                elif rand_float >= 0.1: #assertive
                    self.deviation = 1.0
                else: #aggressive
                    self.deviation = 0.5
            case "easy":
                if rand_float >= 0.2: #passive
                    self.deviation = 1.5
                elif rand_float >= 0.05: #assertive
                    self.deviation = 1.0
                else: #aggressive
                    self.deviation = 0.5
        if self.deviation == 1.5: # setting up how many counter offers can be made
            self.maxCounters = 4
        elif self.deviation == 1.0:
            self.maxCounters = 3
        elif self.deviation == 0.5:
            self.maxCounters = 1

        self.totalCounters = 0
        self.proposal = Trader.generateProposal()

    def generateProposal():
        total = temp = random.randint(2, 5)
        recieving, offering = [None] * 3, [None] * 3
        for i in range(2):
            recieving[i] = random.randint(0, temp)
            temp -= recieving[i]
        recieving[2] = temp
        temp = total
        for i in range(2):
            offering[i] = random.randint(0, temp)
            temp -= offering[i]
        offering[2] = temp
        # print([recieving, offering])
        return [recieving, offering] # trader is recieving x, for y

    def getProposal(self):
        return self.proposal
    
    def recieveProposal(self, counter): # 0 = accept, 1 = reject + new offer, 2 = reject + stop trading
        if self.totalCounters == self.maxCounters:
            return 2
        
        recieving, offering = sum(counter[0]), sum(counter[1])
        ogRecieve, ogOffer = sum(self.proposal[0]), sum(self.proposal[1])
        if recieving > offering:
            return 0
        elif recieving >= ogRecieve - self.deviation and offering <= ogOffer + self.deviation:
            return 0
        else:
            self.totalCounters += 1
            self.proposal = Trader.generateProposal()
            return 1
