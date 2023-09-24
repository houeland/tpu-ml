# Load a text file from disk and create mini-batches for training a byte-based Transformer model.

import jax
import jax.numpy as jnp
from typing import Iterator, NamedTuple

VOCAB_SIZE = 256
SEQUENCE_LENGTH = 256
BATCH_SIZE = 16


class Batch(NamedTuple):
    inputs: jax.Array
    targets: jax.Array


def load_from_file(filename: str) -> Iterator[Batch]:
    with open(filename, "rb") as file:
        array = jnp.frombuffer(file.read(), dtype=jnp.uint8)

    crop_len = SEQUENCE_LENGTH + 1
    num_batches, remainder = jnp.divmod(array.shape[0], BATCH_SIZE * crop_len)
    if remainder:
        array = array[:-remainder]
    ds = array.reshape([-1, crop_len])
    it = iter(ds)

    batch = []
    for row in it:
        batch.append(row)
        if len(batch) == BATCH_SIZE:
            data = jnp.stack(batch)
            yield Batch(inputs=data[:, :-1], targets=data[:, 1:])
            batch = []
