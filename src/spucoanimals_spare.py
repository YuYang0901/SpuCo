import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.optim import SGD

from spuco.datasets import SpuCoAnimals
from spuco.evaluate import Evaluator
from spuco.group_inference import SpareInference
from spuco.invariant_train import SpareTrain
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/spucoanimals_spare.csv")

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_lr", type=float, default=1e-3)
parser.add_argument("--infer_weight_decay", type=float, default=1e-4)
parser.add_argument("--infer_momentum", type=float, default=0.9)
parser.add_argument("--infer_num_epochs", type=int, default=1)

parser.add_argument("--high_sampling_power", type=int, default=2)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
)
trainset.initialize()

valset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    transform=transform,
)
valset.initialize()

testset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    transform=transform,
)
testset.initialize()

print(trainset.group_partition.keys())

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

# if args.pretrained or args.infer_num_epochs > -1:
if 0:
    trainer = Trainer(
        trainset=trainset,
        model=model,
        batch_size=args.batch_size,
        optimizer=SGD(model.parameters(), lr=args.infer_lr, weight_decay=args.infer_weight_decay, momentum=args.infer_momentum),
        device=device,
        verbose=True
    )

    trainer.train(num_epochs=args.infer_num_epochs)

    logits = trainer.get_trainset_outputs()
    predictions = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    spare_infer = SpareInference(
        Z=predictions,
        class_labels=trainset.labels,
        device=device,
        high_sampling_power=args.high_sampling_power,
        verbose=True
    )

    group_partition, sampling_powers = spare_infer.infer_groups()

else:
    max_f1 = np.zeros(trainset.num_classes)
    sampling_powers = np.zeros(trainset.num_classes)
    group_partition = {}
    # logits_files = glob(f"logits/spucoanimals/lr=0.001_wd=0.0001_seed={args.seed}/trainset*.pt")
    # for logits_file in logits_files:
    #     epoch = int(logits_file.split("/")[-1].split(".")[0].split("_")[-1])
    #     logits = torch.load(logits_file)
    trainer = Trainer(
        trainset=trainset,
        model=model,
        batch_size=args.batch_size,
        optimizer=SGD(model.parameters(), lr=args.infer_lr, weight_decay=args.infer_weight_decay, momentum=args.infer_momentum),
        device=device,
        verbose=True
    )
    for epoch in range(args.infer_num_epochs):
        trainer.train(num_epochs=1)
        logits = trainer.get_trainset_outputs()
        predictions = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        spare_infer = SpareInference(
            Z=predictions,
            class_labels=trainset.labels,
            max_clusters=2,
            device=device,
            high_sampling_power=args.high_sampling_power,
            verbose=True
        )
        epoch_group_partition, epoch_sampling_powers = spare_infer.infer_groups()

        for key in sorted(epoch_group_partition.keys()):
            print("Inferred group: {}, size: {}".format(key, len(epoch_group_partition[key])))

        for key in sorted(epoch_group_partition.keys()):
            for true_key in sorted(trainset.group_partition.keys()):
                if len([x for x in trainset.group_partition[true_key] if x in epoch_group_partition[key]]) > 0:
                    print("Inferred group: {}, true group: {}, size: {}".format(key, true_key, len([x for x in trainset.group_partition[true_key] if x in epoch_group_partition[key]])))

        for class_label in np.unique(trainset.labels):
            class_indices = [i for i in range(len(trainset.labels)) if trainset.labels[i] == class_label]
            print("Class: {}, size: {}".format(class_label, len(class_indices)))
            if len(epoch_group_partition[(class_label,1)]) < len(epoch_group_partition[(class_label,0)]):
                upsampled_indices = epoch_group_partition[(class_label,1)].copy()
            else:
                upsampled_indices = epoch_group_partition[(class_label,0)].copy()
            # if len(epoch_group_partition[(1,0)]) < len(epoch_group_partition[(1,1)]):
            #     upsampled_indices.extend(epoch_group_partition[(1,0)].copy())
            # else:
            #     upsampled_indices.extend(epoch_group_partition[(1,1)].copy())
            # if len(epoch_group_partition[(2,1)]) < len(epoch_group_partition[(2,0)]):
            #     upsampled_indices.extend(epoch_group_partition[(2,1)].copy())
            # else:
            #     upsampled_indices.extend(epoch_group_partition[(2,0)].copy())
            # if len(epoch_group_partition[(3,0)]) < len(epoch_group_partition[(3,1)]):
            #     upsampled_indices.extend(epoch_group_partition[(3,0)].copy())
            # else:
            #     upsampled_indices.extend(epoch_group_partition[(3,1)].copy())
            if class_label < 2:
                if len(trainset.group_partition[(class_label,1)]) < len(trainset.group_partition[(class_label,0)]):
                    minority_indices = trainset.group_partition[(class_label,1)].copy()
                else:
                    minority_indices = trainset.group_partition[(class_label,0)].copy()
            else:
                if len(trainset.group_partition[(class_label,3)]) < len(trainset.group_partition[(class_label,2)]):
                    minority_indices = trainset.group_partition[(class_label,3)].copy()
                else:
                    minority_indices = trainset.group_partition[(class_label,2)].copy()
            # minority_indices = trainset.group_partition[(0,1)].copy()
            # minority_indices.extend(trainset.group_partition[(1,0)].copy())
            # minority_indices.extend(trainset.group_partition[(2,3)].copy())
            # minority_indices.extend(trainset.group_partition[(3,2)].copy())
            # compute F1 score on the validation set

            upsampled = np.zeros(len(predictions))
            upsampled[np.array(upsampled_indices)] = 1
            minority = np.zeros(len(predictions))
            minority[np.array(minority_indices)] = 1
            # f1 = f1_score(minority, upsampled)
            f1 = f1_score(minority[class_indices], upsampled[class_indices])
            if f1 > max_f1[class_label]:
                max_f1[class_label] = f1
                args.infer_num_epochs = epoch
                spare_infer = SpareInference(
                    Z=predictions,
                    class_labels=trainset.labels,
                    max_clusters=5,
                    device=device,
                    high_sampling_power=args.high_sampling_power,
                    verbose=True
                )
                epoch_group_partition, epoch_sampling_powers = spare_infer.infer_groups()
                for key in sorted(epoch_group_partition.keys()):
                    if key[0] == class_label:
                        group_partition[key] = epoch_group_partition[key].copy()
                sampling_powers[class_label] = epoch_sampling_powers[class_label]
                print("New best F1 score:", f1, "at epoch", epoch)


print("Sampling powers: {}".format(sampling_powers))
for key in sorted(group_partition.keys()):
    for true_key in sorted(trainset.group_partition.keys()):
        print("Inferred group: {}, true group: {}, size: {}".format(key, true_key, len([x for x in trainset.group_partition[true_key] if x in group_partition[key]])))

evaluator = Evaluator(
    testset=trainset,
    group_partition=group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()

# reinstantiate the model
model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
valid_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=valset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=train_transform,
)
trainset.initialize()

spare_train = SpareTrain(
    model=model,
    num_epochs=args.num_epochs,
    trainset=trainset,
    group_partition=group_partition,
    sampling_powers=sampling_powers,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    val_evaluator=valid_evaluator,
    verbose=True
)
spare_train.train()

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["seed"] = args.seed
results["high_sampling_power"] = args.high_sampling_power
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size
results["infer_num_epochs"] = args.infer_num_epochs
results["infer_lr"] = args.infer_lr
results["infer_weight_decay"] = args.infer_weight_decay

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=spare_train.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()

results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)