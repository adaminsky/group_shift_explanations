ARG BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM ${BASE_IMAGE} as conda

# Install conda dependences
RUN conda install -y numpy pandas
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

RUN mkdir explainable_domain_shift

# Copying library and data for experiments
COPY src explainable_domain_shift/src
COPY setup.py explainable_domain_shift/setup.py
COPY scripts explainable_domain_shift/scripts
COPY data/adult explainable_domain_shift/data/adult
COPY data/nlp explainable_domain_shift/data/nlp
COPY data/breast_cancer explainable_domain_shift/data/breast_cancer
COPY data/imagenetx/imagenet_x_val_multi_factor.jsonl explainable_domain_shift/data/imagenetx/imagenet_x_val_multi_factor.jsonl
COPY data/breeds explainable_domain_shift/data/breeds
COPY data/imagenetx/source_captions.pkl explainable_domain_shift/data/imagenetx/source_captions.pkl
COPY data/imagenetx/target_captions.pkl explainable_domain_shift/data/imagenetx/target_captions.pkl

WORKDIR explainable_domain_shift

# Install packages
RUN python -m pip install POT geomloss scikit-learn scikit-image dice-ml opencv-python wilds transformers jupyterlab notebook
RUN python -m pip install -U git+https://github.com/MadryLab/robustness.git
RUN python -m pip install -U scikit-learn scikit-image; \
    python -m pip install -e .

WORKDIR scripts

CMD ["python", "base_exp.py"]