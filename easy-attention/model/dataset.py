# Load a text file from disk and create mini-batches for training a byte-based Transformer model.

import jax
import jax.numpy as jnp
from typing import Iterator, NamedTuple

from configs import default as config_lib

config = config_lib.get_config()

VOCAB_SIZE = 256


class Batch(NamedTuple):
    inputs: jax.Array
    targets: jax.Array


def load_from_file(filename: str) -> Iterator[Batch]:
    with open(filename, "rb") as file:
        array = jnp.frombuffer(file.read(), dtype=jnp.uint8)

    crop_len = config.dataset_sequence_length + 1  # type: ignore
    _num_batches, remainder = jnp.divmod(
        array.shape[0], config.dataset_batch_size * crop_len
    )
    if remainder:
        array = array[:-remainder]
    ds = array.reshape([-1, crop_len])
    it = iter(ds)

    batch = []
    for row in it:
        batch.append(row)
        if len(batch) == config.dataset_batch_size:
            data = jnp.stack(batch)
            yield Batch(inputs=data[:, :-1], targets=data[:, 1:])
            batch = []
