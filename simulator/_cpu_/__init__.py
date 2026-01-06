from . import clock

class CPU:
    def __init__(self, pre_emptive):
        self.idle_time = 0
        self.pre_emptive = pre_emptive
        self.clock = clock.register()
        pass

    def reset(self):
        self.idle_time = 0
        self.clock.reset()
        pass

def initialize(pre_emptive) -> CPU:
    return CPU(pre_emptive)