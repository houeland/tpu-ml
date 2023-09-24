from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
import pprint

from arrays import WrappedArray, wrap


class Transformer(nn.Module):
    num_attention_heads: int
    num_layers: int  # each layer has attention + MLP
    attention_size_per_head: int
    dropout_rate: float
    is_training: bool
    widening_factor: int = 4  # factor for widening MLP hidden layer

    @nn.compact
    def __call__(self, embeddings: WrappedArray) -> WrappedArray:
        initializer = nn.initializers.variance_scaling(
            2 / self.num_layers, "fan_in", "truncated_normal"
        )
        num_batches, seq_len, embedding_size = embeddings.shape
        print(f"{num_batches=} {seq_len=} {embedding_size=}")

        # Compute causal mask for auto-regressive model
        causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))
        print(f"{causal_mask=}")

        h = embeddings.jax_array
        for _ in range(self.num_layers):
            # First the attention block.
            attn_block = nn.SelfAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.attention_size_per_head * self.num_attention_heads,
                dropout_rate=self.dropout_rate,
                kernel_init=nn.initializers.variance_scaling(
                    scale=2 / self.num_layers,
                    mode="fan_in",
                    distribution="truncated_normal",
                ),
                # use built-in normalization?
                # use_bias False???
                # broadcast_dropout False???
                # use decode:true for speedup!
            )
            h_norm = nn.LayerNorm()(h)
            h_attn = attn_block(
                inputs_q=h_norm, mask=causal_mask, deterministic=not self.is_training
            )
            # add another dropout layer here??? instead of the built-in one?
            h = h + h_attn

            # Then the MLP block
            h_norm = nn.LayerNorm()(h)
            h_dense = nn.Dense(
                self.widening_factor * embedding_size, kernel_init=initializer
            )(h_norm)
            # relu+dropout instead of gelu??
            h_dense = jax.nn.gelu(h_dense)
            h_dense = nn.Dense(embedding_size, kernel_init=initializer)(h_dense)
            h_dense = nn.Dropout(
                rate=self.dropout_rate, deterministic=not self.is_training
            )(h_dense)
            h = h + h_dense

        return nn.LayerNorm()(h)


class AutoregressiveTransformerModel(nn.Module):
    transformer: Transformer
    embed_dim: int
    vocab_size: int
    is_training: bool

    @nn.compact
    def __call__(self, in_tokens: jax.Array) -> jax.Array:
        """Forward pass, producing a sequence of logits."""
        # tokens.shape = (BATCH_SIZE, SEQUENCE_LENGTH)
        tokens = wrap(in_tokens)
        print(f"{tokens=}")
        num_batches, seq_len = tokens.shape
        print(f"{num_batches=}, {seq_len=}")

        # token_embeddings.shape = (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE)
        token_embeddings = wrap(
            nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.embed_dim,
                embedding_init=jax.nn.initializers.truncated_normal(stddev=0.02),
                name="embed__model_tokens_input",
            )(tokens.jax_array)
        )
        # positional_embeddings.shape = (SEQUENCE_LENGTH, EMBEDDING_SIZE)
        # alternatively, fixed sinusoidal embeddings, or rotary embeddings?
        positional_embeddings = wrap(
            self.param(
                "positional_embeddings",
                lambda key: jax.nn.initializers.truncated_normal(stddev=0.02)(
                    key, [seq_len, self.embed_dim]
                ),
            )
        )
        print(f"{token_embeddings=}, {positional_embeddings=}")

        # input_embeddings.shape = (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE)
        # add dropout???
        input_embeddings = wrap(
            token_embeddings.jax_array + positional_embeddings.jax_array
        )
        # output.shape = ???
        rng_key = jax.random.key(12345)
        output = wrap(self.transformer(input_embeddings))
        print(f"{input_embeddings=}, {output=}")

        decoded = wrap(
            nn.Dense(
                self.vocab_size,
                name="dense__decode_transformer_output",
            )(output.jax_array)
        )
        print(f"{decoded=}")
        return decoded.jax_array


if __name__ == "__main__":
    import dataset

    model = AutoregressiveTransformerModel(
        transformer=Transformer(
            num_attention_heads=3,
            num_layers=2,
            attention_size_per_head=4,
            dropout_rate=0.1,
            is_training=False,
        ),
        embed_dim=9,
        vocab_size=dataset.VOCAB_SIZE,
        is_training=False,
    )
    print(model)
    rng = jax.random.key(12345)
    init_rng, apply_rng = jax.random.split(rng)
    data = dataset.Batch(
        inputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        targets=jnp.array([[2, 3, 4, 5], [6, 7, 8, 9]]),
    )
    # print(model.tabulate(init_rng, data.inputs))
    variables = jax.jit(model.init)(init_rng, data.inputs)
    pprint.pp(["variables", jax.tree_util.tree_map(lambda a: a.shape, variables)])
    output = jax.jit(model.apply)(
        variables,
        rngs={"dropout": apply_rng},
        in_tokens=data.inputs,
    )
    pprint.pp(["output", jax.tree_util.tree_map(lambda a: a.shape, output)])
