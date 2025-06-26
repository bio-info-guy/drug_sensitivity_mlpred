# Use the specified base image
FROM --platform=linux/amd64 nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (corrected architecture)
RUN wget -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Set PATH for conda commands globally for subsequent RUN commands
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment (fixed environment name)
RUN mamba create -n work python=3.11 -y

# Install PyTorch, CUDA toolkit, and Jupyter into the new environment
RUN mamba install -y -n work \
    pytorch \
    torchvision \
    torchaudio \
    jupyter \
    jupyterlab \
    ipykernel \
    imbalanced-learn \
    lightgbm \
    xgboost \
    numpy \
    scipy \
    matplotlib \
    optuna \
    optuna-integration \
    cuml \
    pandas \
    scikit-learn \
    seaborn \
    -c pytorch \
    -c rapidsai \
    -c nvidia \
    -c conda-forge && \
    conda clean -afy

# Install pip packages in the correct environment
RUN /opt/conda/envs/work/bin/pip install names_generator sklearn-evaluation

# Set environment variables for the container runtime
ENV PATH="/opt/conda/envs/work/bin:/opt/conda/bin:$PATH"
ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV CONDA_DEFAULT_ENV="work"

# Define metadata labels
LABEL Author="Yangqi Su" \
      Version="v1.0" \
      Description="PyTorch / ML container"

# Expose Jupyter port
EXPOSE 8888

# Define the default command to run when the container starts
CMD ["bash"]