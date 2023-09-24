import dataclasses
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class WrappedArray:
    jax_array: jax.Array

    @property
    def shape(self):
        return self.jax_array.shape

    def __str__(self):
        return str(self.jax_array)

    def __repr__(self):
        num_entries = 1
        for v in self.shape:
            num_entries *= v
        if num_entries > 20:
            flattened = self.jax_array.reshape(-1)
            return f"shape={self.shape} dtype={self.jax_array.dtype} values=[{flattened[0]}, {flattened[1]}, {flattened[2]}, ...]"
        else:
            return f"shape={self.shape} dtype={self.jax_array.dtype}   " + str(
                self.jax_array
            )


def wrap(jax_array):
    return WrappedArray(jax_array)
