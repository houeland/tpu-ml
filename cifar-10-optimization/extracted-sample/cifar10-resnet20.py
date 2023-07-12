import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="BATCH_SIZE to use to run the model")
parser.add_argument("--peak_value", help="PEAK_VALUE to use for learning rate")
parser.add_argument(
    "--use_nesterov", help="USE_NESTEROV setting to use for learning rate"
)
parser.add_argument("--lr_decay", help="LR_DECAY setting to use for learning rate")
parser.add_argument(
    "--augmentation",
    help="AUGMENTATION setting to use for training batch preprocessing",
)
parser.add_argument("--prng_seed", help="PRNG_SEED to use for this run")
args = parser.parse_args()

BATCH_SIZE = int(args.batch_size)  # or 1024
PEAK_VALUE = float(args.peak_value)  # or 0.1
USE_NESTEROV = args.use_nesterov == "true"
LR_DECAY = float(args.lr_decay)  # or 0.9
PRNG_SEED = int(args.prng_seed)  # or 42
augs = args.augmentation.split(",")
USE_RANDOM_CROP = "randomcrop" in augs
USE_FLIP_LEFTRIGHT = "fliplr" in augs
USE_GLOBAL_NORMALIZATION = "normalize" in augs

SHUFFLE_SIZE = 2048
EPOCHS = 100

# DEFAULT: batch=128 and lr_schedule = optax.cosine_onecycle_schedule(transition_steps=100*50000//128, peak_value=0.1)

# import sys
# sys.exit(0)

# pip install "jax[tpu]>=0.2.21" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install jax==0.3.17 jaxlib==0.3.15 dm-haiku==0.0.8 optax==0.1.3 tensorflow-datasets==4.6.0 tensorflow==2.9.1

from typing import Any, NamedTuple, Tuple, List
import time
import functools
import json
import sys

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

jax_utils__Batch = jnp.ndarray

def jax_utils__eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def jax_utils__initialize_environment():
    jax.config.update("jax_numpy_rank_promotion", "raise")
    eprint("JAX version", jax.__version__)
    eprint("JAX devices:", [d.device_kind for d in jax.devices()])
    try:
        import tensorflow as tf

        eprint("TensorFlow version", tf.__version__)
    except:
        pass
    try:
        import tensorflow_datasets as tfds

        eprint("TensorFlow Datasets version", tfds.__version__)
    except:
        pass
    try:
        import haiku as hk

        eprint("Haiku version", hk.__version__)
    except:
        pass


CIFAR10_MEAN = jnp.array([0.485, 0.456, 0.406])
CIFAR10_STD = jnp.array([0.229, 0.224, 0.225])
def tfds_utils__normalize_cifar10_images(images):
    return (images - jnp.broadcast_to(CIFAR10_MEAN, images.shape)) / jnp.broadcast_to(
        CIFAR10_STD, images.shape
    )


class jax_utils__HkBasicBlock(hk.Module):
    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        name=None,
    ):
        super().__init__(name=name)
        self.in_planes = in_planes
        self.conv1 = hk.Conv2D(
            output_channels=planes,
            kernel_shape=3,
            stride=stride,
            padding=(1, 1),
            with_bias=False,
        )
        self.bn1 = hk.BatchNorm(
            create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-05
        )
        self.conv2 = hk.Conv2D(
            output_channels=planes,
            kernel_shape=3,
            stride=1,
            padding=(1, 1),
            with_bias=False,
        )
        self.bn2 = hk.BatchNorm(
            create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-05
        )

        self.shortcut = hk.Sequential([])
        if stride != 1 or in_planes != planes:
            self.shortcut = lambda x: jnp.pad(
                x[:, ::2, ::2, :], ((0, 0), (0, 0), (0, 0), (planes // 4, planes // 4))
            )

    def __call__(
        self,
        inputs: jnp.ndarray,
        is_training,
    ) -> jnp.ndarray:
        assert (
            inputs.shape[-1] == self.in_planes
        ), f"inputs.shape={inputs.shape} self.in_planes={self.in_planes}"
        x = inputs
        out = x
        out = self.conv1(out)
        out = self.bn1(out, is_training=is_training)
        out = jax.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, is_training=is_training)
        out += self.shortcut(x)
        out = jax.nn.relu(out)
        return out


def jax_utils__hk_make_resnet20(x, is_training):
    this_batch_size = x.shape[0]

    latest_self_in_planes = [16]

    out = x
    out = hk.Conv2D(
        output_channels=16,
        kernel_shape=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        with_bias=False,
    )(out)
    out = hk.BatchNorm(
        create_scale=True, create_offset=True, decay_rate=0.9, eps=1e-05
    )(out, is_training)
    out = jax.nn.relu(out)
    assert out.shape == (this_batch_size, 32, 32, 16), f"out.shape={out.shape}"

    def make_layer(block, planes, num_blocks, stride):
        # print("make_layer()")
        def run(x, is_training):
            strides = [stride] + [1] * (num_blocks - 1)
            for iter_stride in strides:
                x = block(latest_self_in_planes[0], planes, iter_stride)(
                    x, is_training=is_training
                )
                latest_self_in_planes[0] = planes
                # print("latest_self_in_planes", latest_self_in_planes)
            return x

        return run

    out = make_layer(jax_utils__HkBasicBlock, planes=16, num_blocks=3, stride=1)(
        out, is_training=is_training
    )
    assert out.shape == (this_batch_size, 32, 32, 16), f"out.shape={out.shape}"

    out = make_layer(jax_utils__HkBasicBlock, planes=32, num_blocks=3, stride=2)(
        out, is_training=is_training
    )
    assert out.shape == (this_batch_size, 16, 16, 32), f"out.shape={out.shape}"

    out = make_layer(jax_utils__HkBasicBlock, planes=64, num_blocks=3, stride=2)(
        out, is_training=is_training
    )
    assert out.shape == (this_batch_size, 8, 8, 64), f"out.shape={out.shape}"

    # torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) → Tensor
    # out = F.avg_pool2d(out, out.size()[3])
    # haiku.avg_pool(value, window_shape, strides, padding, channel_axis=- 1)
    out = hk.avg_pool(
        out, window_shape=(1, 8, 8, 1), strides=(1, 8, 8, 1), padding="SAME"
    )
    assert out.shape == (this_batch_size, 1, 1, 64), f"out.shape={out.shape}"

    out = hk.Flatten()(out)
    assert out.shape == (this_batch_size, 64), f"out.shape={out.shape}"

    out = hk.Linear(output_size=10, with_bias=True)(out)
    assert out.shape == (this_batch_size, 10), f"out.shape={out.shape}"

    return out


def jax_utils__count_trainable_parameters(xs):
    leaves = jax.tree_leaves(xs)
    shapes = [l.shape for l in leaves]
    return sum(jnp.product(jnp.array(s)) for s in shapes)


#from jax_utils import eprint
eprint = jax_utils__eprint
jax_utils__initialize_environment()

eprint(
    f"BATCH_SIZE={BATCH_SIZE} PEAK_VALUE={PEAK_VALUE} USE_NESTEROV={USE_NESTEROV} LR_DECAY={LR_DECAY} PRNG_SEED={PRNG_SEED} USE_RANDOM_CROP={USE_RANDOM_CROP} USE_FLIP_LEFTRIGHT={USE_FLIP_LEFTRIGHT} USE_GLOBAL_NORMALIZATION={USE_GLOBAL_NORMALIZATION} EPOCHS={EPOCHS} SHUFFLE_SIZE={SHUFFLE_SIZE}"
)


def test_hk_make_resnet20():
    rng_key = jax.random.PRNGKey(42)
    hk_model = hk.transform_with_state(jax_utils__hk_make_resnet20)
    params, state = hk_model.init(
        rng=rng_key, x=jnp.zeros((128, 32, 32, 3)), is_training=True
    )
    num_trainable_weights = jax_utils__count_trainable_parameters(params)
    assert num_trainable_weights == 269722
    eprint("num_trainable_weights", num_trainable_weights)


# test_hk_make_resnet20()


PERCENTILES_RECORDED = jnp.array((0, 10, 25, 50, 75, 90, 100))


@functools.partial(jax.jit, static_argnums=(0, 1))
def crossentropy_loss_fn(
    model, is_training, params: hk.Params, state, batch: jax_utils__Batch
):
    this_batch_size = batch["src_images"].shape[0]
    preds, new_state = model.apply(
        params, state, None, batch["src_images"], is_training=is_training
    )
    # print(f"preds.shape={preds.shape}")
    assert preds.shape == (this_batch_size, 10)
    one_hot_labels = jax.nn.one_hot(batch["src_labels"], 10)
    batch_loss = optax.softmax_cross_entropy(preds, one_hot_labels)
    assert batch_loss.shape == (this_batch_size,)
    # print(f"batch_loss.shape={batch_loss.shape}")
    return jnp.mean(batch_loss), (new_state, batch_loss)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def update(
    loss_fn,
    optimizer,
    model,
    params: hk.Params,
    state,
    opt_state: optax.OptState,
    batch: jax_utils__Batch,
):
    grads, (new_state, batch_loss) = jax.grad(
        lambda *x: loss_fn(model, True, *x), has_aux=True
    )(params, state, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return (
        new_params,
        new_state,
        new_opt_state,
        dict(batch_loss_percentiles=jnp.percentile(batch_loss, PERCENTILES_RECORDED)),
    )


def test_stuff():
    rng_key = jax.random.PRNGKey(42)
    model = hk.transform_with_state(jax_utils__hk_make_resnet20)
    optimizer = optax.adamw(0.001)
    params, state = model.init(
        rng=rng_key, x=jnp.zeros((128, 32, 32, 3)), is_training=True
    )
    opt_state = optimizer.init(params)
    batch = next(list_augmented_cifar10_train_batches())
    params, state, opt_state, extra_stats = update(
        crossentropy_loss_fn, optimizer, model, params, state, opt_state, batch
    )
    batch_loss_percentiles = extra_stats["batch_loss_percentiles"]
    eprint(f"batch_loss_percentiles={batch_loss_percentiles}")

    loss, _ = crossentropy_loss_fn(model, False, params, state, batch)
    eprint(f"loss={loss}")


# test_stuff()


@functools.partial(jax.jit, static_argnums=0)
def get_model_predictions(model, params: hk.Params, state, batch: jax_utils__Batch):
    targets = batch["src_labels"]
    preds, new_state = model.apply(
        params, state, None, batch["src_images"], is_training=False
    )
    return targets, preds


def list_cifar10_test_batches():
    ds = tfds.load("cifar10")
    for x in ds["test"].cache().batch(BATCH_SIZE).as_numpy_iterator():
        image = x["image"]
        image = jnp.array(image) / 255.0
        if USE_GLOBAL_NORMALIZATION:
            image = tfds_utils__normalize_cifar10_images(image)
        yield dict(src_images=image, src_labels=x["label"])


def check_test_accuracy(model, params, state):
    good = 0
    total = 0
    for batch in list_cifar10_test_batches():
        targets, preds = get_model_predictions(model, params, state, batch)
        predicted_labels = jnp.argmax(preds, axis=1)
        correct = jnp.count_nonzero(targets == predicted_labels)
        good += int(correct)
        total += batch["src_images"].shape[0]
    return good / total


class Record(NamedTuple):
    num_processed_images: int
    validation_batch_loss: float
    #  train_accuracy: float
    test_accuracy: float
    epoch: int
    train_duration_secs: float
    eval_duration_secs: float


class BatchRecord(NamedTuple):
    num_processed_images: int
    batch_loss: List[float]


def list_augmented_cifar10_train_batches():
    ds = tfds.load("cifar10")
    for x in (
        ds["train"].cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).as_numpy_iterator()
    ):
        image = x["image"]
        if USE_RANDOM_CROP:
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, (image.shape[0], 32, 32, 3))
        if USE_FLIP_LEFTRIGHT:
            image = tf.image.random_flip_left_right(image)
        image = jnp.array(image) / 255.0
        if USE_GLOBAL_NORMALIZATION:
            image = tfds_utils__normalize_cifar10_images(image)
        yield dict(src_images=image, src_labels=x["label"])


records = []
batch_records = []

run_training_start_time = time.monotonic()

model = hk.transform_with_state(jax_utils__hk_make_resnet20)

lr_schedule = optax.cosine_onecycle_schedule(
    transition_steps=EPOCHS * 50000 // BATCH_SIZE, peak_value=PEAK_VALUE
)

optimizer = optax.chain(
    optax.trace(decay=LR_DECAY, nesterov=USE_NESTEROV),
    optax.scale_by_schedule(lambda count: -lr_schedule(count)),
)

rng_key = jax.random.PRNGKey(PRNG_SEED)

rng_key, rng_model_init, rng_batch = jax.random.split(rng_key, 3)
params, state = model.init(
    rng=rng_model_init, x=jnp.zeros((128, 32, 32, 3)), is_training=True
)
# print("params", debug_tree_shape(params))
opt_state = optimizer.init(params)

training_start_time = time.monotonic()

validation_batch = next(list_cifar10_test_batches())
eprint("starting training...")
num_processed_images = 0
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.monotonic()
    for training_batch in list_augmented_cifar10_train_batches():
        params, state, opt_state, extra_stats = update(
            crossentropy_loss_fn,
            optimizer,
            model,
            params,
            state,
            opt_state,
            training_batch,
        )
        batch_loss = extra_stats["batch_loss_percentiles"]
        num_processed_images += training_batch["src_images"].shape[0]
        batch_records.append(
            BatchRecord(
                num_processed_images=num_processed_images,
                batch_loss=batch_loss.tolist(),
            )
        )

    epoch_train_end_time = time.monotonic()
    validation_batch_loss, _ = crossentropy_loss_fn(
        model, False, params, state, validation_batch
    )
    test_accuracy = check_test_accuracy(model, params, state)
    epoch_end_time = time.monotonic()
    eprint(
        f"epoch={epoch} num_processed_images={num_processed_images} epoch={epoch}: validation_batch_loss = {validation_batch_loss:.5f}"
    )
    eprint("  epoch duration", epoch_end_time - epoch_start_time)
    eprint("  train duration", epoch_train_end_time - epoch_start_time)
    eprint("  eval duration", epoch_end_time - epoch_train_end_time)
    # print("  train_accuracy", train_accuracy)
    eprint("  test_accuracy ", test_accuracy)
    records.append(
        Record(
            num_processed_images=num_processed_images,
            validation_batch_loss=validation_batch_loss.tolist(),
            # train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            epoch=epoch,
            train_duration_secs=epoch_train_end_time - epoch_start_time,
            eval_duration_secs=epoch_end_time - epoch_train_end_time,
        )
    )

run_training_end_time = time.monotonic()

time_elapsed = dict(
    total=run_training_end_time - run_training_start_time,
    training=run_training_end_time - training_start_time,
    setup=training_start_time - run_training_start_time,
)
eprint(time_elapsed)

print(
    json.dumps(
        dict(
            time_elapsed=time_elapsed,
            records=[r._asdict() for r in records],
            batch_records=[br._asdict() for br in batch_records],
        )
    )
)
