ARG BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM ${BASE_IMAGE} as conda

# Install conda dependences
RUN conda install -y numpy pandas
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

RUN mkdir group_shift_explanations

# Copying library and data for experiments
COPY src group_shift_explanations/src
COPY setup.py group_shift_explanations/setup.py
COPY scripts group_shift_explanations/scripts

WORKDIR group_shift_explanations

# Install packages
RUN python -m pip install POT geomloss scikit-learn scikit-image dice-ml opencv-python wilds transformers jupyterlab notebook clip-interrogator
RUN python -m pip install -U git+https://github.com/MadryLab/robustness.git
RUN python -m pip install -U scikit-learn scikit-image; \
    python -m pip install -e .

WORKDIR scripts

CMD ["python", "robustness_exp.py"]
