import torch.optim as optim
import torch.nn as nn
import torch

from model import VCNN


def main():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(VCNN.parameters(), lr=0.001, momentum=0.9)

    torch.cuda.set_device(0)

    # model training
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = VCNN(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


if __name__ == '__main__':
    main()
