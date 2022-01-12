import logging

from dask.distributed import Client
from nvtabular.utils import _pynvml_mem_size, device_mem_size


def init_cluster(
    dask_workdir,
    processes=True,
    protocol="tcp",
    enable_tcp_over_ucx=False,
    enable_infiniband=False,
    enable_nvlink=False,
    enable_rdmacm=False,
    ucx_net_devices=None,
    rmm_pool_size=None,
    rmm_managed_memory=False,
    threads_per_worker=1,
    dashboard_port="8787",
):
    from dask_cuda import LocalCUDACluster

    try:
        import torch

        NUM_GPUS = [i for i in range(torch.cuda.device_count())]
    except ImportError:
        import tensorflow as tf

        NUM_GPUS = [int(_gpu.name.split(":")[-1]) for _gpu in tf.config.get_visible_devices("GPU")]

    visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Delect devices to place workers
    device_limit_frac = 0.7  # Spill GPU-Worker memory to host at this limit.

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)

    # TODO: figure out what these are
    # device_pool_frac = 0.8
    # part_mem_frac = 0.15
    # device_pool_size = int(device_pool_frac * device_size)
    # part_size = int(part_mem_frac * device_size)

    # Check if any device memory is already occupied
    for dev in visible_devices.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            logging.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    cluster = LocalCUDACluster(
        protocol=protocol,
        n_workers=len(visible_devices.split(",")),
        CUDA_VISIBLE_DEVICES=visible_devices,
        device_memory_limit=device_limit,
        local_directory=dask_workdir,
        dashboard_address=":" + dashboard_port,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        enable_rdmacm=enable_rdmacm,
        ucx_net_devices=ucx_net_devices,
        rmm_pool_size=rmm_pool_size,
        rmm_managed_memory=rmm_managed_memory,
        threads_per_worker=threads_per_worker,
        processes=processes,
    )
    return cluster, Client(cluster)
