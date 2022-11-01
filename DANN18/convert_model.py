import torch


model = torch.load("models/mnist_mnistm_model_epoch_current.pth")
torch.save(model.state_dict(), "models/mnist_mnistm_model_epoch_current_dict.pth")
