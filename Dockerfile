FROM continuumio/miniconda3:24.7.1-0

COPY environment.yml .
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "conda activate gis_env" >> ~/.bashrc
ENV PATH="$PATH:/opt/conda/envs/gis_env/bin"

# Create a non-root user and switch to that user
RUN useradd -m gisuser
USER gisuser

WORKDIR /home/gisuser

# Expose the ports for jupyter lab and dask
EXPOSE 8888

EXPOSE 8787

# Start jupiter lab
CMD ["jupyter", "lab", "--ip=0.0.0.0"]

