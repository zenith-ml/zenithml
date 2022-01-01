def init_hvd():
    import cupy
    import tf
    import horovod.tensorflow as hvd
    from horovod.tensorflow import mpi_ops

    cupy.random.seed(None)
    hvd.init()

    def seed_fn():
        """
        Generate consistent dataloader shuffle seeds across workers
        Reseeds each worker's dataloader each epoch to get fresh a shuffle
        that's consistent across workers.
        """
        min_int, max_int = tf.int32.limits
        max_rand = max_int // hvd.size()

        # Generate a seed fragment on each worker
        seed_fragment = cupy.random.randint(0, max_rand).get()

        # Aggregate seed fragments from all Horovod workers
        seed_tensor = tf.constant(seed_fragment)
        reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=mpi_ops.Sum)

        return reduced_seed % max_rand

    return hvd, seed_fn
