# syntax=docker/dockerfile:1
FROM jupyter/scipy-notebook

USER root

# Update
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends ffmpeg dvipng cm-super && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Dependencies
RUN conda install pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

USER ${NB_UID}

# Run Jupyter Notebook
ENTRYPOINT ["jupyter", "notebook"]
CMD ["--port=8888", "--no-browser",\
    "--ip=0.0.0.0", "--allow-root"]

EXPOSE 8888