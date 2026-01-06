import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "_data_")
TRAINING_DATA_PATH = os.path.join(DATA_PATH, "training_data", "episodes.pkl")
TESTING_DATA_PATH = os.path.join(DATA_PATH, "testing_data", "episodes.pkl")
CHECKPOINT_PATH = os.path.join(DATA_PATH, "model_parameter.pth")
RESULTS_PLOT_PATH = os.path.join(BASE_PATH, "results", "graph.png")
RESULTS_LOG_PATH = os.path.join(BASE_PATH, "results", "logs.txt")

def trim_path(path):
    return os.path.relpath(path, BASE_PATH)
