from typing import Callable, Optional, List

import ray
from ray.autoscaler.sdk import get_head_node_ip

from zenithml.utils import rich_logging


def init_ray(config_file, py_modules=None, working_dir="."):
    head_node_ip = get_head_node_ip(config_file)
    logger = rich_logging()
    logger.info(f"Using head node: {head_node_ip}")
    logger.info(f"Dashboard: http://{head_node_ip}:8265")
    logger.info(f"Jupyter Notebook: http://{head_node_ip}:7777 (password:p)")
    ray.init(
        f"ray://{head_node_ip}:10001",
        runtime_env={"py_modules": py_modules, "working_dir": working_dir},
        namespace="zenithml",
    )


def runner(py_modules: Optional[List] = None, working_dir=".", logging_level=None, **ray_kwargs):
    def runner_wrapper(func: Callable):
        def main(cluster_config: Optional[str] = None, detach: bool = False, *args, **kwargs):

            assert cluster_config, "cluster-config must be passed as argument"
            init_ray(cluster_config, py_modules, working_dir)
            _ray_kwargs = {"num_cpus": 1} if ray_kwargs == {} else ray_kwargs

            @ray.remote
            class RunnerActor:
                def task(self):
                    import logging

                    logging.getLogger().setLevel(logging_level or logging.INFO)
                    logging.root.setLevel(logging.INFO)
                    logging.basicConfig(level=logging.INFO)
                    print(logging.root.level)
                    return func(*args, **kwargs)

            if detach:
                runner_actor = RunnerActor.options(  # type: ignore
                    name="RunnerActor", lifetime="detached", **_ray_kwargs
                ).remote()
                runner_actor.task.remote()
            else:
                runner_actor = RunnerActor.options(name="RunnerActor", **_ray_kwargs).remote()  # type: ignore
                ray.get(runner_actor.task.remote())
                ray.kill(runner_actor)

        return main

    return runner_wrapper
