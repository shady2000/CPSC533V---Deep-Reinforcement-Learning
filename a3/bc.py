import gym
import torch
from eval_policy import eval_policy, device
from model import MyModel
from dataset import Dataset
import torch.optim as optim
import torch.nn as nn

BATCH_SIZE = 64
TOTAL_EPOCHS = 100
LEARNING_RATE = 10e-4
PRINT_INTERVAL = 500
TEST_INTERVAL = 2

ENV_NAME = 'CartPole-v0'

dataset = Dataset(data_path="D:/Machine Learning/FPT AI Residency/CPSC533V_2021W1/a3/{}_dataset.pkl".format(ENV_NAME))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

env = gym.make(ENV_NAME)

# TODO INITIALIZE YOUR MODEL HERE
model = MyModel(4, 2)


def train_behavioral_cloning():
    
    # TODO CHOOSE A OPTIMIZER AND A LOSS FUNCTION FOR TRAINING YOUR NETWORK
    optimizer = optim.RMSprop(model.parameters(), lr=1e-2, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    gradient_steps = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        for iteration, data in enumerate(data_loader):
            # data loader: contains data. data.items() --> dictionary
            data = {k: v.to(device) for k, v in data.items()}

            output = model(data['state'])

            loss = loss_function(output, data["action"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gradient_steps % PRINT_INTERVAL == 0:
                print('[epoch {:4d}/{}] [iter {:7d}] [loss {:.5f}]'
                    .format(epoch, TOTAL_EPOCHS, gradient_steps, loss.item()))
            
            gradient_steps += 1

        if epoch % TEST_INTERVAL == 0:
            score = eval_policy(policy=model, env=ENV_NAME)
            print('[Test on environment] [epoch {}/{}] [score {:.2f}]'
                .format(epoch, TOTAL_EPOCHS, score))

    model_name = "behavioral_cloning_{}.pt".format(ENV_NAME)
    print('Saving model as {}'.format(model_name))
    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    train_behavioral_cloning()
