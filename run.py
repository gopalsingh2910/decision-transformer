import decision_transformer as DT

from utils.loader import generate_data, load_model, save_model
from utils.paths import TRAINING_DATA_PATH, TESTING_DATA_PATH
from decision_transformer import param as p

if __name__ == "__main__":
    model = load_model()
    # generate_data(epoches_num=p.EPISODES_NUM*p.TRAINING_FRACTION, path=TRAINING_DATA_PATH)
    # generate_data(epoches_num=p.EPISODES_NUM*p.TESTING_FRACTION, path=TESTING_DATA_PATH)
    # DT.train(model, TRAINING_DATA_PATH, int(p.EPISODES_NUM*p.TRAINING_FRACTION))
    # save_model(model, path=None)
    # score = DT.test(model, TESTING_DATA_PATH, int(p.EPISODES_NUM*p.TESTING_FRACTION))
    DT.run(model)
    pass