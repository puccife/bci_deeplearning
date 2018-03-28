from model.cnn.train_cnn import CNNTrainer
from model.cnn.basic_cnn import CNN

net = CNN()

def run():
    CNNTrainer(cnn=net).train_cnn()

if __name__ == "__main__":
    run()