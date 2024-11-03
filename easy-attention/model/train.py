import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.warning("start")

import jax
import jax.experimental
import jax.numpy as jnp
import ml_collections
import optax
from flax.training import train_state

if False:
    from flax.metrics import tensorboard
else:
    tensorboard = None

import model
import dataset


class TrainState(train_state.TrainState):
    num_examples_trained_on: int


def create_model(config: ml_collections.ConfigDict):
    return model.AutoregressiveTransformerModel(
        transformer=model.Transformer(
            num_attention_heads=config.transformer_num_attention_heads,  # type: ignore
            num_layers=config.transformer_num_layers,  # type: ignore
            attention_size_per_head=config.transformer_attention_size_per_head,  # type: ignore
            dropout_rate=config.transformer_dropout_rate,  # type: ignore
            is_training=True,
        ),
        embed_dim=config.model_embed_dim,
        vocab_size=dataset.VOCAB_SIZE,
        is_training=True,
    )


def create_train_state(
    init_rng: jax.random.PRNGKeyArray, config: ml_collections.ConfigDict
):
    m = create_model(config)
    shape = (config.dataset_batch_size, config.dataset_sequence_length)
    data = dataset.Batch(
        inputs=jnp.zeros(shape, dtype=jnp.uint8),
        targets=jnp.ones(shape, dtype=jnp.uint8),
    )
    params_rng_key, dropout_rng_key = jax.random.split(init_rng)
    variables = jax.jit(m.init)(
        {"params": params_rng_key, "dropout": dropout_rng_key}, data.inputs
    )
    tx = optax.sgd(config.learning_rate, config.momentum)  # type: ignore
    return TrainState.create(
        apply_fn=m.apply, params=variables["params"], tx=tx, num_examples_trained_on=0
    )


def log_progress(num_examples_trained_on, loss, accuracy):
    logging.info(f"[{num_examples_trained_on}] loss: {loss}, accuracy: {accuracy}")


@jax.jit
def apply_model(state, batch, apply_rng_key):
    dropout_rng_key = apply_rng_key

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, batch.inputs, rngs={"dropout": dropout_rng_key}
        )
        log_probs = jax.nn.log_softmax(logits)  # [B, T, V]
        onehot_targets = jax.nn.one_hot(batch.targets, dataset.VOCAB_SIZE)
        log_likelihood = jnp.sum(onehot_targets * log_probs, axis=-1)  # [B, T]

        # Loss is the average negative log-likelihood per (non-masked) token.
        return -jnp.mean(log_likelihood), {"logits": logits}

    def importance_fn(inputs):
        logits = state.apply_fn(
            {"params": state.params}, inputs, rngs={"dropout": dropout_rng_key}
        )
        # ...

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(aux["logits"], -1) == batch.targets)
    jax.experimental.io_callback(
        log_progress, None, state.num_examples_trained_on, loss, accuracy
    )
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train(
    rng_key: jax.random.PRNGKeyArray, config: ml_collections.ConfigDict, workdir: str
):
    if tensorboard:
        summary_writer = tensorboard.SummaryWriter(workdir)
        summary_writer.hparams(dict(config))
    init_rng, rng_key = jax.random.split(rng_key)
    state = create_train_state(init_rng, config)
    print(f"{state=}")

    for batch in dataset.load_from_file("data/generated-easy-attention-dataset.txt"):
        rng_key, apply_rng_key = jax.random.split(rng_key)
        grads, loss, accuracy = apply_model(state, batch, apply_rng_key)
        state = update_model(state, grads)
        state = state.replace(
            num_examples_trained_on=state.num_examples_trained_on
            + batch.inputs.shape[0]
        )


if __name__ == "__main__":
    from configs import default as config_lib
    config = config_lib.get_config()
    train_rng = jax.random.key(12345)
    workdir = "./training_workdir/"
    train(train_rng, config, workdir)


#     # Create train state with Adam optimizer and weight decay.
#     learning_rate_fn = create_learning_rate_schedule(
#         learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
#     )
