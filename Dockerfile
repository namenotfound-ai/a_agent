# Use an official Miniconda3 base image
FROM continuumio/miniconda3:latest

# Install build essentials, including gcc, g++, and make
# This runs as root by default in the miniconda3 image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    # Add other system dependencies if known to be needed by any Python packages
    # For example, llama-cpp-python might benefit from BLAS libraries if compiling from source
    # for specific CPU optimizations, though its wheels are usually pre-compiled.
    # cmake \ # If llama-cpp-python or other packages need it to build
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the Docker-specific requirements file
COPY requirements.docker.txt ./requirements.txt

# Create the Conda environment
ARG PYTHON_VERSION=3.10
RUN conda create -n dyn_agent_env python=${PYTHON_VERSION} -y && \
    conda init bash && \
    echo "conda activate dyn_agent_env" >> ~/.bashrc

# Set the SHELL to run subsequent commands within the Conda environment
SHELL ["conda", "run", "-n", "dyn_agent_env", "/bin/bash", "-c"]

# Install pip packages from requirements.txt
# Now gcc should be available for packages like blis/thinc
RUN pip install -r requirements.txt

# Install supervisor and gunicorn
RUN pip install supervisor gunicorn

# Conda clean to reduce image size
RUN conda clean -afy

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy application files into /app
COPY aggregator.py .
COPY index.html .
COPY settings_orgs.json .

# Copy the entire Models directory to /app/Models
COPY ./Models ./Models

# Create and set permissions for directories used by the apps
RUN mkdir -p /app/workspaces_v1_5 && chmod -R 777 /app/workspaces_v1_5
RUN mkdir -p /app/Models/context_store && chmod -R 777 /app/Models/context_store

# Expose the ports your applications will run on
EXPOSE 8000
EXPOSE 5035

# Command to run Supervisor, which will start your services
CMD ["/opt/conda/envs/dyn_agent_env/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]