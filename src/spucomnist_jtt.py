import argparse
import os

import pandas as pd
import torch
from torch.optim import SGD

from spuco.datasets import (GroupLabeledDatasetWrapper, SpuCoMNIST,
                            SpuriousFeatureDifficulty)
from spuco.evaluate import Evaluator
from spuco.group_inference import JTTInference
from spuco.invariant_train import CustomSampleERM
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")

parser.add_argument("--arch", type=str, default="lenet")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_num_epochs", type=int, default=1)

parser.add_argument("--upsample_factor", type=int, default=800)

parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--feature_noise", type=float, default=0.0)
parser.add_argument("--feature_difficulty", type=SpuriousFeatureDifficulty, default=SpuriousFeatureDifficulty.MAGNITUDE_LARGE,
                    choices=list(SpuriousFeatureDifficulty))

args = parser.parse_args()

args.results_csv = f'results/spucomnist_jtt_{args.feature_difficulty}.csv'

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
difficulty = args.feature_difficulty

trainset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    core_feature_noise=args.feature_noise,
    label_noise=args.label_noise,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train"
)
trainset.initialize()

# # save group partition
# import pickle

# os.makedirs('group_partition', exist_ok=True)
# with open(f'group_partition/trainset.pkl', 'wb') as f:
#     pickle.dump(trainset.group_partition, f)

valset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="val",
    #spurious_correlation_strength=0.995
)
valset.initialize()

testset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test"
)
testset.initialize()

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
trainer.train(num_epochs=args.infer_num_epochs)

predictions = torch.argmax(trainer.get_trainset_outputs(), dim=-1).detach().cpu().tolist()
jtt = JTTInference(
    predictions=predictions,
    class_labels=trainset.labels
)

group_partition = jtt.infer_groups()

# save group partition
import pickle

os.makedirs('group_partition', exist_ok=True)
with open(f'group_partition/jtt_{difficulty}.pkl', 'wb') as f:
    pickle.dump(group_partition, f)

import sys
sys.exit()

for key in sorted(group_partition.keys()):
    print(key, len(group_partition[key]))
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

invariant_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

indices = []
indices.extend(group_partition[(0,0)])
indices.extend(group_partition[(0,1)] * args.upsample_factor)

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
val_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
jtt_train = CustomSampleERM(
    model=model,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    indices=indices,
    val_evaluator=val_evaluator,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
jtt_train.train()

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
results["pretrained"] = args.pretrained
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size
results["infer_num_epochs"] = args.infer_num_epochs
results["upsample_factor"] = args.upsample_factor
results["difficulty"] = difficulty
results['label_noise'] = args.label_noise
results['feature_noise'] = args.feature_noise

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=jtt_train.best_model,
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


