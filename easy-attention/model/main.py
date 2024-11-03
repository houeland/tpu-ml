import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.warning("start")

import jax
import sys
from train import train
from configs import default as config_lib

config = config_lib.get_config()


def main():
    # print("hello, world!")
    # ds = dataset.load_from_file("generated-easy-attention-dataset.txt")
    # for idx, batch in enumerate(ds):
    #     print(idx, batch)
    # print("super down")
    train_rng = jax.random.key(12345)
    workdir = "./training_workdir/"
    train(train_rng, config, workdir)


if __name__ == "__main__":
    sys.exit(main())
