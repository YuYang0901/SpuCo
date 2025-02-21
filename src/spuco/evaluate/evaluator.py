import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from spuco.datasets import SpuriousTargetDatasetWrapper
from spuco.utils.random_seed import seed_randomness

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class Evaluator:
    def __init__(
        self,
        testset: Dataset, 
        group_partition: Dict[Tuple[int, int], List[int]],
        group_weights: Dict[Tuple[int, int], float],
        batch_size: int,
        model: nn.Module,
        sklearn_linear_model: Optional[Tuple[float, float, float, Optional[StandardScaler]]] = None,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
        eval_aiou: bool = False
    ):
        """
        Initializes an instance of the Evaluator class.

        :param testset: Dataset object containing the test set.
        :type testset: Dataset

        :param group_partition: Dictionary object mapping group keys to a list of indices corresponding to the test samples in that group.
        :type group_partition: Dict[Tuple[int, int], List[int]]

        :param group_weights: Dictionary object mapping group keys to their respective weights.
        :type group_weights: Dict[Tuple[int, int], float]

        :param batch_size: Batch size for DataLoader.
        :type batch_size: int

        :param model: PyTorch model to evaluate.
        :type model: nn.Module

        :param sklearn_linear_model: Tuple representing the coefficients and intercept of the linear model from sklearn. Default is None.
        :type sklearn_linear_model: Optional[Tuple[float, float, float, Optional[StandardScaler]]], optional

        :param device: Device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional

        :param verbose: Whether to print evaluation results. Default is False.
        :type verbose: bool, optional
        :param eval_aiou: Whether to evaluate the average intersection over union (IoU) for each group. Default is False.
        :type eval_aiou: bool, optional
        """

        if eval_aiou:
            assert 'mask_dict' in testset.__dict__.keys(), 'mask_dict not found in testset'
          
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.testloaders = {}
        self.group_partition = group_partition
        self.group_weights = group_weights
        self.model = model
        self.device = device
        self.verbose = verbose
        self.accuracies = None
        self.sklearn_linear_model = sklearn_linear_model
        self.n_classes = np.max(testset.labels) + 1
        self.eval_aiou = eval_aiou

        # Create DataLoaders 

        # Group-Wise DataLoader
        for key in group_partition.keys():
            sampler = SubsetRandomSampler(group_partition[key])
            self.testloaders[key] = DataLoader(testset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, shuffle=False)
        
        # SpuriousTarget Dataloader
        core_labels = []
        spurious = torch.zeros(len(testset))
        for key in self.group_partition.keys():
            for i in self.group_partition[key]:
                spurious[i] = key[1]
                core_labels.append(key[0])
        spurious_dataset = SpuriousTargetDatasetWrapper(dataset=testset, spurious_labels=spurious, num_classes=np.max(core_labels) + 1)
        self.spurious_dataloader = DataLoader(spurious_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    def evaluate(self):
        """
        Evaluates the PyTorch model on the test dataset and computes the accuracy for each group.
        """
        self.model.eval()
        self.accuracies = {}
        self.aiou = {}
        for key in tqdm(sorted(self.group_partition.keys()), "Evaluating group-wise accuracy", ):
            if self.sklearn_linear_model:
                self.accuracies[key] = self._evaluate_accuracy_sklearn_logreg(self.testloaders[key])
            else:
                self.accuracies[key] = self._evaluate_accuracy(self.testloaders[key])
            
            if self.eval_aiou:
                self.aiou[key] = self._evaluate_aiou(self.testloaders[key])

            if self.verbose:
                print(f"Group {key} Accuracy: {self.accuracies[key]}")
                if self.eval_aiou:
                    print(f"Average IoU: {np.mean(list(self.aiou.values()))}")
        
        if self.eval_aiou:
            return self.accuracies, self.aiou
        else:
            return self.accuracies
    
    def _evaluate_accuracy(self, testloader: DataLoader):
        with torch.no_grad():
            correct = 0
            total = 0    
            for batch in testloader:
                inputs, labels = batch[0], batch[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            return 100 * correct / total
    
    def _evaluate_accuracy_sklearn_logreg(self, testloader: DataLoader):
        C, coef, intercept, scaler = self.sklearn_linear_model

        X_test, y_test = self._encode_testset(testloader)
        X_test = X_test.detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        if scaler:
            X_test = scaler.transform(X_test)
        logreg = LogisticRegression(penalty='l1', C=C, solver="liblinear")
        # the fit is only needed to set up logreg
        X_dummy = np.random.rand(self.n_classes, X_test.shape[1])
        logreg.fit(X_dummy, np.arange(self.n_classes))
        logreg.coef_ = coef
        logreg.intercept_ = intercept
        preds_test = logreg.predict(X_test)
        return (preds_test == y_test).mean()

    def _evaluate_aiou(self, testloader: DataLoader):
        with torch.no_grad():
            aious = []
            for batch in testloader:
                inputs, labels, masks = batch[0], batch[1], batch[2]
                inputs, labels, masks = inputs.to(self.device), labels.to(self.device), masks.to(self.device)
                
                # get the gradcam mask
                gradcam_mask = self.model.get_gradcam_mask(inputs, labels)
                gradcam_mask = gradcam_mask.detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                for i in range(len(gradcam_mask)):
                    aious.append(self._compute_aiou(gradcam_mask[i], masks[i]))
            return np.mean(aious)
        
    def _compute_aiou(self, mask1: np.ndarray, mask2: np.ndarray, label: int = 0) -> float:
        """
        Computes the intersection over union (IoU) for the given masks adjusted for the given label.

        :param mask1: The first mask.
        :type mask1: np.ndarray
        :param mask2: The second mask.
        :type mask2: np.ndarray
        :param label: The label to compute the IoU for. Default is 0.
        :type label: int, optional
        :return: The Adjusted IoU.
        :rtype: float
        """

        # compute the intersection by taking the minimum of the two masks
        intersection = np.minimum(mask1, mask2)
        # compute the union by taking the maximum of the two masks
        union = np.maximum(mask1, mask2)

        # compute the IoU
        iou = np.sum(intersection[label]) / np.sum(union[label])

        # adjust the IoU by dividing it by the sum of iou of the label and the highest iou of the other labels
        adjusted_iou = iou / (iou + np.max([np.sum(intersection[i]) / np.sum(union[i]) for i in range(len(union)) if i != label]))

        return adjusted_iou

    
    def _encode_testset(self, testloader):
        X_test = []
        y_test = []

        self.model.eval()
        with torch.no_grad():
            for batch in testloader:
                inputs, labels = batch[0], batch[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                X_test.append(self.model.backbone(inputs))
                y_test.append(labels)
            return torch.cat(X_test), torch.cat(y_test)
        
    def evaluate_spurious_attribute_prediction(self):
        """
        Evaluates accuracy if the task was predicting the spurious attribute.
        """
        return self._evaluate_accuracy(self.spurious_dataloader)

    @property
    def worst_group_accuracy(self):
        """
        Returns the group with the lowest accuracy and its corresponding accuracy.

        :returns: A tuple containing the key of the worst-performing group and its corresponding accuracy.
        :rtype: tuple
        """
        if self.accuracies is None:
            print("Run evaluate() first")
            return None
        else:
            min_key = min(self.accuracies, key=self.accuracies.get)
            min_value = min(self.accuracies.values())
            return (min_key, min_value)
    
    @property
    def average_accuracy(self):
        """
        Returns the weighted average accuracy across all groups.

        :returns: The weighted average accuracy across all groups.
        :rtype: float
        """
        if self.accuracies is None:
            print("Run evaluate() first")
            return None
        else:
            accuracy = 0
            for key in self.group_partition.keys():
                accuracy += self.group_weights[key] * self.accuracies[key]
            return accuracy