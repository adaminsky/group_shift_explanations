# Group-aware Shift Explanations
This is the official repository for the paper [Rectifying Group Irregularities in Explanations for Distribution Shift](https://arxiv.org/abs/2305.16308).

Before running any scripts, download the necessary datasets into the `data/`
directory by following the directions under "Dataset Setup." The scripts can
then be run using Docker as described in Section "Running with Docker."

## Dataset Setup
### ImageNet Datset
Download the ImageNet dataset from
https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description.
If you are not using Docker, then create a soft link to the directory
`ILSVRC/Data/CLS-LOC/val/` in the ImageNet data to the directory
`data/imagenet/` in this repository.

## Running with Docker
Navigate to the main directory `group_shift_explanations/` and build the docker
image with the following:
```bash
docker build -t explainshift:latest .
```

### Reproducing experiments from the paper
To rerun all the the experiments shown in the paper, run the following:

Robustness experiments:
```bash
docker run -it \
	-v $(pwd)/data:/workspace/group_shift_explanations/data \
	explainshift:latest \
	python robustness_exp.py --method kmeans --dataset adult --train
```

Worst-case Robustness experiments:
```bash
docker run -it \
	-v $(pwd)/data:/workspace/group_shift_explanations/data \
	explainshift:latest \
	python robustness_exp.py --method kmeans --dataset adult --train --adv
```

The allowed values for the `--method` flag are `kmeans` for K-cluster, `dice`,
and `ot`. The allowed values for the `--dataset` flag are `adult`, `breast`,
`nlp`, and `imagenet`.

Regular experiments:
```bash
# For the CivilComments experiments
docker run -it \
	-v $(pwd)/data:/workspace/group_shift_explanations/data \
	explainshift:latest \
	python nlp_explore.py
```

```bash
# For the ImageNet and Tabular data experiments
docker run -it \
	-v $(pwd)/data:/workspace/group_shift_explanations/data \
	-v <PATH-TO-IMAGENET>/ILSVRC/Data/CLS-LOC/val/:/workspace/group_shift_explanations/data/imagenet \
	-p 8888:8888 explainshift:latest \
	/bin/bash -c "jupyter lab --ip 0.0.0.0 --allow-root --no-browser"
```

The provided scripts with their experiments are listed below:

- `tabular_explore_multiseed.ipynb`: Tabular data experiments
- `imagenet_breed_groups_multiseed.ipynb`: ImageNet experiments
- `nlp_explore.py`: Language data experiments

### Using a custom dataset
To learn a Group-aware Shift Explanation on a custom dataset, use the
`scripts/custom_data.ipynb` notebook. This notebook takes an arbitrary source
and target dataset along with groups for the source and target data and then
learns a Group-aware Shift Explanation. Follow the instructions for "Running
with Docker" to run the jupyter lab server and then open the
`scripts/custom_data.ipynb` notebook.

# Citation
```bibtex
@article{stein2023rectifying,
  title={Rectifying Group Irregularities in Explanations for Distribution Shift},
  author={Stein, Adam and Wu, Yinjun and Wong, Eric and Naik, Mayur},
  journal={arXiv preprint arXiv:2305.16308},
  year={2023}
}
```