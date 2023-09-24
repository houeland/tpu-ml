import jax
import jax.numpy as jnp
import haiku as hk

import model
import dataset

MODEL_EMBED_DIM = 4

TRANSFORMER_NUM_ATTENTION_HEADS = 4
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_ATTENTION_SIZE = 32
TRANSFORMER_DROPOUT_RATE = 0.1


def forward_pass(tokens: jax.Array) -> jax.Array:
    lm = model.AutoregressiveTransformerModel(
        embed_dim=MODEL_EMBED_DIM,
        vocab_size=dataset.VOCAB_SIZE,
        transformer=model.Transformer(
            num_attention_heads=TRANSFORMER_NUM_ATTENTION_HEADS,
            num_layers=TRANSFORMER_NUM_LAYERS,
            attention_size=TRANSFORMER_ATTENTION_SIZE,
            dropout_rate=TRANSFORMER_DROPOUT_RATE,
        ),
    )
    return lm(tokens)


@hk.transform
def loss_fn(data: dataset.Batch) -> jax.Array:
    """Computes the (scalar) language modelling loss on `data` w.r.t. params."""
    logits = forward_pass(data.inputs)
    log_probs = jax.nn.log_softmax(logits)  # [B, T, V]
    onehot_targets = jax.nn.one_hot(data.targets, dataset.VOCAB_SIZE)
    log_likelihood = jnp.sum(onehot_targets * log_probs, axis=-1)  # [B, T]

    # Loss is the average negative log-likelihood per (non-masked) token.
    return -jnp.sum(log_likelihood)


if __name__ == "__main__":
    data = dataset.Batch(
        inputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        targets=jnp.array([[2, 3, 4, 5], [6, 7, 8, 9]]),
    )
    rng = jax.random.PRNGKey(12345)
    init_rng, apply_rng = jax.random.split(rng)
    initial_params = loss_fn.init(init_rng, data)
    output = loss_fn.apply(initial_params, apply_rng, data)
    print(output)
