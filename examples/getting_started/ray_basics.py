import logging
import socket
import time
from collections import Counter

logging.getLogger().setLevel(logging.DEBUG)
import ray

ray.init("ray://127.0.0.1:10001")


@ray.remote
def f():
    time.sleep(0.001)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())


object_ids = [f.remote() for _ in range(10000)]
ip_addresses = ray.get(object_ids)

print("Tasks executed")
for ip_address, num_tasks in Counter(ip_addresses).items():
    print("    {} tasks on {}".format(num_tasks, ip_address))
