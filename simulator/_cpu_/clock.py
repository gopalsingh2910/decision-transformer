class Clock:
    def __init__(self):
        self.tick = 0
        pass
    
    def reset(self):
        self.tick = 0
        pass

    def step(self):
        self.tick += 1
        pass

def register() -> Clock:
    return Clock()