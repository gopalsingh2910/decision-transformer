import matplotlib.pyplot as plt

from utils.paths import RESULTS_PLOT_PATH

class Plot:
    def __init__(self):
        self.rewards_transformer = []
        self.rewards_fcfs = []
        self.rewards_sjf = []
        self.rewards_rr = []
        self.rewards_srtf = []
        pass

    def log_rewards(self, reward_transformer, reward_fcfs, reward_sjf, reward_rr, reward_srtf):
        self.rewards_transformer.append(reward_transformer)
        self.rewards_fcfs.append(reward_fcfs)
        self.rewards_sjf.append(reward_sjf)
        self.rewards_rr.append(reward_rr)
        self.rewards_srtf.append(reward_srtf)
        pass

    def plot_rewards(self):
        print("Plotting rewards...")
        for i in range(1, len(self.rewards_transformer)):
            self.rewards_transformer[i] += self.rewards_transformer[i - 1]
        for i in range(1, len(self.rewards_fcfs)):
            self.rewards_fcfs[i] += self.rewards_fcfs[i - 1]
        for i in range(1, len(self.rewards_sjf)):
            self.rewards_sjf[i] += self.rewards_sjf[i - 1]
        for i in range(1, len(self.rewards_rr)):
            self.rewards_rr[i] += self.rewards_rr[i - 1]
        for i in range(1, len(self.rewards_srtf)):
            self.rewards_srtf[i] += self.rewards_srtf[i - 1]
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.rewards_transformer, label='Transformer', color='blue')
        plt.plot(self.rewards_fcfs, label='FCFS', color='orange')
        plt.plot(self.rewards_sjf, label='SJF', color='green')
        plt.plot(self.rewards_rr, label='RR', color='red')
        plt.plot(self.rewards_srtf, label='SRTF', color='purple')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title('Rewards over Time for Different Scheduling Policies')
        plt.legend()
        plt.grid(True)
        plt.savefig(RESULTS_PLOT_PATH)
        print("Plotting completed.")
        pass