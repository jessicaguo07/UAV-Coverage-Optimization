from Hyperparameters import Hyperparameters
from Algorithm import DQL

if __name__ == '__main__':
    map_size = (30,30)
    hyperparameters = Hyperparameters(map_size = map_size)
    #hyperparameters.change(map_size=map_size,epsilon_decay=.99)
    
    #train = True
    train = False
    # Run
    DRL = DQL(hyperparameters, train_mode=train) # Define the instance
    # Train
    if train:
        DRL.train()
    else:
        DRL.play()