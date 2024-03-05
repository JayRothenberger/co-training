# taken here
# https://github.com/Jonathan-Pearce/calibration_library/blob/master/recalibration.py

import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torch.distributed as dist

import metrics

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.to(logits.get_device())
        return logits / temperature

    def predict(self, world_size, device, loader, batch_size, num_classes):
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for X, y in loader:
                tensor_list = [torch.full((batch_size, num_classes), -1,
                                        dtype=X.dtype).to(device) 
                                for _ in range(world_size)]
                output = self.model(X.to(device))

                # pad the output so that it contains the same 
                # number of rows as specified by the batch size
                pad = torch.full((batch_size, num_classes), -1, 
                                dtype=X.dtype).to(device)
                pad[:output.shape[0]] = output

                # same as above but we need the labels
                label_list = [torch.full((batch_size,), -1, 
                                     dtype=output.dtype).to(device)
                                     for _ in range(world_size)]

                pad2 = torch.full((batch_size,), -1, 
                               dtype=output.dtype).to(device)
                pad2[:y.shape[0]] = y

                # all-gather the full list of predictions across all processes
                dist.all_gather(tensor_list, pad)
                batch_outputs = torch.cat(tensor_list)

                # all-gather the full list of labels
                dist.all_gather(label_list, pad2)
                batch_labels = torch.cat(label_list)

                # remove all rows of the tensor that contain a -1
                # (as this is not a valid value anywhere)
                mask = ~(batch_outputs == -1).any(-1)
                batch_outputs = batch_outputs[mask]
                predictions.append(batch_outputs)

                batch_labels = batch_labels[mask]
                labels.append(batch_labels)

        return torch.cat(predictions), torch.cat(labels)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, world_size, device, valid_loader, batch_size, num_classes):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        #self.cuda()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = metrics.ECELoss()

        # First: collect all the logits and labels for the validation set
        # logits_list = []
        # labels_list = []
        # with torch.no_grad():
        #     for input, label in valid_loader:
        #         input = input
        #         logits = self.model(input)
        #         logits_list.append(logits)
        #         labels_list.append(label)
        #     logits = torch.cat(logits_list)
        #     labels = torch.cat(labels_list)
        logits, labels = self.predict(world_size, device, valid_loader, batch_size, num_classes)
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
        labels_np = labels.detach().cpu().numpy().astype(np.int64)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits.type(torch.float64), labels.type(torch.int64)).item()
        before_temperature_ece = ece_criterion.loss(logits_np, labels_np, 25)
        #before_temperature_ece = ece_criterion(logits, labels).item()
        #ece_2 = ece_criterion_2.loss(logits,labels)
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        #print(ece_2)
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=1e-3, max_iter=500)
        self.temperature.data = torch.clamp(self.temperature.data, -2**16, 2**16)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits).type(torch.float64), labels.type(torch.int64))
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits).type(torch.float64), labels.type(torch.int64)).item()
        after_temperature_ece = ece_criterion.loss(self.temperature_scale(logits).detach().cpu().numpy(),labels.cpu().numpy(),25)
        #after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
