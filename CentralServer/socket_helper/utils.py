import torch as th
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import syft as sy

class Util:

    def __init__(self, grid, model, model_name, model_version):
        self.grid = grid
        self.model_name = model_name
        self.model_version = model_version
        self.model = model
        # Checkpoint
        self.model_accuracies = list()
        self.class_recall = list()
        self.class_precision = list()
        self.confusion_matrix = list()
        self.classification_report = list()
        self.model_checkpoint = dict()
        # dataset 
        self.load_test_data()
    
    def load_test_data(self):
        self.test_dataset = DataLoader(
            datasets.FashionMNIST("./test_dataset", train=False, download=True, transform=transforms.ToTensor()),
            batch_size=64,
            shuffle=True,
            drop_last=True
        )

    def model_validation(self, checkpoint):
        # Model evaluation
        self.model.eval()

        # Variables for check
        correct = 0
        total = 0
        targets_ = []
        pred_ = []

        # Get the model params from PyGrid and inject it into the current model
        model_params = self.grid.retrieve_model("mnist", "1.0.0", 'latest')
        Util.set_model_params(self.model, model_params)

        # Start model validation
        with th.no_grad():
            for data, target in self.test_dataset:
                data, target = data.to(th.device('cpu')), target.to(th.device('cpu'))
                outputs = self.model(data)
                _, predicted = th.max(outputs.data, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

                targets_.extend(target.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())
                #loss += th.nn.CrossEntropyLoss(outputs, target).item()
                #acc = preds.eq(y).float().mean()
                #accuracies.append(acc.item())

        acc = correct / total * 100
        confusion_mat = confusion_matrix(targets_, pred_)
        class_recall = self.calculate_class_recall(confusion_mat)
        class_precision = self.calculate_class_precision(confusion_mat)
        classification_rpt = classification_report(targets_, pred_)
        self.model_accuracies.append(acc)
        self.class_recall.append(class_recall)
        self.class_precision.append(class_precision)
        self.confusion_matrix.append(confusion_mat)
        self.classification_report.append(classification_rpt)
        self.model_checkpoint["model_checkpoint_" + str(checkpoint)] = [
            f"INFORMATION OF MODEL CHECKPOINT {str(checkpoint)}\n",
            f"[MODEL ACCURACY]:\n {str(acc)}",
            f"[CLASSIFICATION REPORT]:\n {classification_rpt}",
            f"[CONFUSION MATRIX]: \n{str(confusion_mat)}",
            f"[CLASS PRECISION]:\n{str(class_precision)}",
            f"[CLASS RECALL]: \n{str(class_recall)}"
        ]
    
    def calculate_class_recall(self, confusion_mat):
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)

    def calculate_class_precision(self, confusion_mat):
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=0)

    def get_confusion_matrix(self, idx):
        if idx == -1:           # Get all
            message = ""
            for idx, cm in enumerate(self.confusion_matrix):
                message += f"[+] Confusion matrix of checkpoint {idx}\n"
                message += str(cm) + "\n"
            return message
        message = f"[+] Confusion matrix of checkpoint {idx}\n"
        message += str(self.confusion_matrix[idx])
        return message

    def get_class_recall(self, idx):
        if idx == -1:           # Get all
            message = ""
            for idx, cr in enumerate(self.class_recall):
                message += f"[+] Class recall of checkpoint {idx}\n"
                message += str(cr) + "\n"
            return message
        message = f"[+] Class recall of checkpoint {idx}\n"
        message += str(self.class_recall[idx])
        return message

    def get_model_accuracy(self, idx):
        if idx == -1:           # Get all
            message = ""
            for idx, ma in enumerate(self.model_accuracies):
                message += f"[+] Model's accuracy of checkpoint {idx}\n"
                message += str(ma) + "\n"
            return message
        message = f"[+] Model's accurary of checkpoint {idx}\n"
        message += str(self.model_accuracies[idx])
        return message

    def get_class_precision(self, idx):
        if idx == -1:           # Get all
            message = ""
            for idx, cp in enumerate(self.class_precision):
                message += f"[+] Class precision of checkpoint {idx}\n"
                message += str(cp) + "\n"
            return message
        message = f"[+] Class precision of checkpoint {idx}\n"
        message += str(self.class_precision[idx])
        return message

    def get_classification_report(self, idx):
        if idx == -1:           # Get all
            message = ""
            for idx, cp in enumerate(self.classification_report):
                message += f"[+] Classification report of checkpoint {idx}\n"
                message += str(cp) + "\n"
            return message
        message = f"[+] Classification report of checkpoint {idx}\n"
        message += str(self.classification_report[idx])
        return message
    
    def get_model_checkpoint(self, idx):
        if idx == -1:           # Get all
            message = ""
            for model_cp, des in self.model_checkpoint:
                message += f"[+] {str(model_cp)}]n"
                message += str(des) + "\n"
            return message
        message = f"[+] Confusion matrix of checkpoint {idx}\n"
        message += str(self.model_checkpoint[idx])
        return message

    @staticmethod
    def set_model_params(model, params):
        for p, p_new in zip(model.parameters(), params):
            p.data = p_new.data
