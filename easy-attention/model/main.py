import sys
import dataset


def main():
    print("hello, world!")
    ds = dataset.load_from_file("generated-easy-attention-dataset.txt")
    for idx, batch in enumerate(ds):
        print(idx, batch)
    print("super down")


if __name__ == "__main__":
    sys.exit(main())
