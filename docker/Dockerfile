ARG RELEASE=false
ARG BASE_VER=21.11
ARG BASE_IMAGE=nvcr.io/nvidia/merlin/merlin-tensorflow-training
FROM $BASE_IMAGE:$BASE_VER
SHELL ["/bin/bash", "-c"]
ENV NVTAB_VER=v0.11.0
ENV HOME /home
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYTHONPATH "${PYTHONPATH}:/nvtabular:$HOME/ml:$HOME/ml/src:$HOME/ml/src:$HOME/ml/lib"

RUN bash -c "$(wget https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh -O -)"
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup  && \
    echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install jupyterlab_nvdashboard dask-labextension

WORKDIR $HOME
ADD docker/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
ADD docker/run_jupyter.sh /run_jupyter.sh
RUN chmod +x /run_jupyter.sh

WORKDIR $HOME/ml
ADD . $HOME/ml
RUN pip install .
#RUN if [[ "$BASE_IMAGE" == *"tensorflow"* ]] ; then pip install .[tf]; else pip install .[torch]; fi
# Clean up
RUN pip install numba==0.53.1 numpy==1.22.2 --no-deps

#EXPOSE 8000-8999
ENTRYPOINT ["/run_jupyter.sh"]
CMD ["/run_jupyter.sh"]