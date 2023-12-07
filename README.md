# Group-aware Shift Explanations
This is the official repository for the paper [Rectifying Group Irregularities in Explanations for Distribution Shift](https://arxiv.org/abs/2305.16308).

Before running any scripts, download the necessary datasets into the `data/`
directory by following the directions under "Dataset Setup." The scripts can
then be run using Docker as described in Section "Running with Docker."

## Dataset Setup
### Adult Dataset
Download the Adult tabular dataset from
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data and
create the directory `data/adult/` where `adult.data` should be placed.

### Breast Dataset
Download the Breast tabular dataset from
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data and create
the directory `data/breast` where `data.csv` should be placed.

### ImageNet Datset
Download the ImageNet dataset from
https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description.
Then, create a soft link to the directory `ILSVRC/Data/CLS-LOC/val/` in the
ImageNet data to the directory `data/imagenet/` in this repository.


## Running with Docker
Navigate to the main directory `explainable_domain_shift/` and build the docker image with the following:
```bash
docker build -t explainshift:latest .
```

To run the experiments in the container, run the following:
```bash
docker run -it explainshift:latest \
	-v $(pwd)/base:/workspace/explainable_domain_shift/data/base \
	python base_exp.py --method kmeans --dataset adult --train
```

Worst-case Robustness experiments:
```bash
docker run -it explainshift:latest \
	-v $(pwd)/base:/workspace/explainable_domain_shift/data/base \
	python base_exp.py --method kmeans --dataset adult --adv --train
```

The allowed values for the `--method` flag are `kmeans` for K-cluster, `dice`,
and `ot`. The allowed values for the `--dataset` flag are `adult`, `breast`,
`nlp`, `nlp-amazon`, `fmow`, `iwildcam`, `imagenet`.


# Citation
```bibtex
@article{stein2023rectifying,
  title={Rectifying Group Irregularities in Explanations for Distribution Shift},
  author={Stein, Adam and Wu, Yinjun and Wong, Eric and Naik, Mayur},
  journal={arXiv preprint arXiv:2305.16308},
  year={2023}
}
```
