# Instructions for training on ImageNet using TensorFlow Datasets

Here are some notes on using the https://www.tensorflow.org/datasets/catalog/imagenet2012 dataset, which is unfortunately non-trivial as of 2023-01-29.

I hope this might be helpful to someone looking to train on ImageNet. The exact steps may not apply to you, but it should give a decent overview of what files and formats are required to use the `imagenet2012` dataset with TFDS.

Big thanks to https://github.com/leondgarse/keras_cv_attention_models/discussions/9 which is where I could find a description of what the data files actually need to look like!

## Downloading on a TPU (which have relatively small disks but a ton of RAM)

- Create a TPUv3 and ssh into the TPU VM.
  - `free -g`
  - ```
                  total        used        free      shared  buff/cache   available
    Mem:            334           1         330           0           2         331
    Swap:             0           0           0
    ```
- Create a ramdisk (which is big enough to hold the ImageNet data, even though the normal disk isn't!):
  - `sudo mkdir /mnt/ramdisk`
  - `sudo mount -t tmpfs -o size=325G tmpfs /mnt/ramdisk`
- Download ImageNet dataset from Kaggle:
  - Note: The exact instruments will vary over time, since Kaggle appears to ~randomly change the format of the available data download files.
  - Find the Download link from somewhere on https://www.kaggle.com/c/imagenet-object-localization-challenge
  - Example #1:
    - This might e.g. be in the form of a link to https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz and get the link, which looks something like https://storage.googleapis.com/kagglesdsdata/competitions/17433/824213/imagenet_object_localization_patched2019.tar.gz?GoogleAccessId=...
    - Download that file onto the TPU VM (~26min at 126MB/s):
      - `cd /mnt/ramdisk/`
      - `time wget -O imagenet_object_localization_patched2019.tar.gz 'https://storage.googleapis.com/kagglesdsdata/competitions/17433/824213/imagenet_object_localization_patched2019.tar.gz?GoogleAccessId=...'`
    - Extract (~19min):
      - `time tar -zxf imagenet_object_localization_patched2019.tar.gz`
      - `rm imagenet_object_localization_patched2019.tar.gz`
  - Example #2:
    - Click "Download All" and find out what the link was.
    - Download that file onto the TPU VM (~24min at 111MB/s):
      - `cd /mnt/ramdisk/`
      - `time wget -O imagenet-object-localization-challenge.zip 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6799/4225553/bundle/archive.zip?GoogleAccessId=...'`
      - ```
        imagenet-object-localization-challenge.zip         100%[================================================================================================================>] 155.06G   120MB/s    in 23m 51s
        
        2022-09-19 10:38:31 (111 MB/s) - ‘imagenet-object-localization-challenge.zip’ saved [166496672546/166496672546]
        
        
        real    23m51.099s
        user    2m30.191s
        sys     3m37.029s
        ```
    - Extract (~19min):
      - `unzip imagenet-object-localization-challenge.zip`
      - `rm imagenet-object-localization-challenge.zip`
- Depending on the downloads available from Kaggle, some other adjustments may be necessary. The results at this point should look like this:
  - ```
    houeland@t1v-n-52e1df7c-w-0:/mnt/ramdisk$ du -chd4
    61M     ./ILSVRC/ImageSets/CLS-LOC
    61M     ./ILSVRC/ImageSets
    6.4G    ./ILSVRC/Data/CLS-LOC/val
    140G    ./ILSVRC/Data/CLS-LOC/train
    13G     ./ILSVRC/Data/CLS-LOC/test
    159G    ./ILSVRC/Data/CLS-LOC
    159G    ./ILSVRC/Data
    196M    ./ILSVRC/Annotations/CLS-LOC/val
    2.1G    ./ILSVRC/Annotations/CLS-LOC/train
    2.3G    ./ILSVRC/Annotations/CLS-LOC
    2.3G    ./ILSVRC/Annotations
    161G    ./ILSVRC
    316G    .
    316G    total
    ```
  - These directories should contain JPEG images, e.g. `ILSVRC/Data/CLS-LOC/train/n01443537/n01443537_10014.JPEG`, `ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_00000002.JPEG`, and `ILSVRC/Data/CLS-LOC/ILSVRC2012_val_00000003.JPEG`.
  - We don't need the ImageSets or Annotations:
    - `rm -r ILSVRC/ImageSets/ ILSVRC/Annotations/`

## Convert ImageNet data to the format TFDS expects

- Next the files need to converted into a different format suitable for use with TFDS:
  - Instead of having individual files such as `ILSVRC/Data/CLS-LOC/train/n01443537/n01443537_10014.JPEG`, we need .tar files that contain the JPEG files inside them. E.g. `n01443537.tar` should contain `n01443537_10014.JPEG` and the other images from the same directory.
  - `cd ILSVRC/Data/CLS-LOC/train`
  - `mkdir ../train-tar-files`
  - `time for n in *; do cd "$n"; echo "$n"; tar cf "../../train-tar-files/$n.tar" *.JPEG; cd ..; done`
  - `cd ../`
  -  ```
     houeland@t1v-n-83ba1784-w-0:/mnt/ramdisk/ILSVRC/Data/CLS-LOC$ du -chd1
     138G	./train-tar-files
     140G	./train
     13G	./test
     6.4G	./val
     297G	.
     297G	total
     ```
- Then these training .tar files need to be bundled together inside a `ILSVRC2012_img_train.tar` file:
  - `rm -r train/`
  - `cd train-tar-files/`
  - `time tar cvf ../ILSVRC2012_img_train.tar *.tar`
  - `cd ..`
  - ```
    houeland@t1v-n-83ba1784-w-0:/mnt/ramdisk/ILSVRC/Data/CLS-LOC$ ls -lh
    total 138G
    -rw-rw-r-- 1 houeland houeland 138G Sep  1 20:27 ILSVRC2012_img_train.tar
    drwxr-xr-x 2 houeland houeland 2.0M May 17  2015 test
    drwxrwxr-x 2 houeland houeland  20K Sep  1 20:11 train-tar-files
    drwxr-xr-x 2 houeland houeland 977K May 17  2015 val
    houeland@t1v-n-83ba1784-w-0:/mnt/ramdisk/ILSVRC/Data/CLS-LOC$ du -chd1
    138G    ./train-tar-files
    13G     ./test
    6.4G    ./val
    295G    .
    295G    total
    ```
  - `rm -r train-tar-files/`
- Next, the test and validation files need to similarly be bundled together inside `ILSVRC2012_img_test.tar` and `ILSVRC2012_img_val.tar` files. (Unlike the training data, these should directly contain JPEG files.)
  - `cd test && ls > ../test-filelist && tar cvf ../ILSVRC2012_img_test.tar -T ../test-filelist && cd ..`
  - `cd val && ls > ../val-filelist && tar cvf ../ILSVRC2012_img_val.tar -T ../val-filelist && cd ..`
  - ```
    houeland@t1v-n-83ba1784-w-0:/mnt/ramdisk/ILSVRC/Data/CLS-LOC$ ls -lh
    total 157G
    -rw-rw-r-- 1 houeland houeland  13G Sep  1 20:37 ILSVRC2012_img_test.tar
    -rw-rw-r-- 1 houeland houeland 138G Sep  1 20:27 ILSVRC2012_img_train.tar
    -rw-rw-r-- 1 houeland houeland 6.3G Sep  1 20:38 ILSVRC2012_img_val.tar
    drwxr-xr-x 2 houeland houeland 2.0M Sep  1 20:36 test
    -rw-rw-r-- 1 houeland houeland 2.9M Sep  1 20:36 test-filelist
    drwxr-xr-x 2 houeland houeland 977K May 17  2015 val
    -rw-rw-r-- 1 houeland houeland 1.4M Sep  1 20:38 val-filelist
    ```
  - `rm -r test test-filelist val val-filelist`

## Place the files where TFDS expects them

Depending on your version of TFDS, the manually downloaded datasets may need to be in a different directory. It's supposed to be `$HOME/tensorflow_datasets/downloads/manual`, but some versions have a bug and instead require a directory literally named `~`, so it would be `./~/tensorflow_datasets/downloads/manual` instead.

- Create a separate directory on the ramdisk to store the processed files:
  - `mkdir -p /mnt/ramdisk/tensorflow_datasets/downloads/manual`
  - `mv *.tar /mnt/ramdisk/tensorflow_datasets/downloads/manual/`
  - `ln -s /mnt/ramdisk/tensorflow_datasets/ ~/`
- The initial dataset download should now be complete:
  - ```
    houeland@t1v-n-83ba1784-w-0:~$ ls -lh ~/tensorflow_datasets/downloads/manual/
    total 157G
    -rw-rw-r-- 1 houeland houeland  13G Sep  1 20:37 ILSVRC2012_img_test.tar
    -rw-rw-r-- 1 houeland houeland 138G Sep  1 20:27 ILSVRC2012_img_train.tar
    -rw-rw-r-- 1 houeland houeland 6.3G Sep  1 20:38 ILSVRC2012_img_val.tar
    ```
  - ```
    houeland@t1v-n-52e1df7c-w-0:~$ find ~/tensorflow_datasets/ -ls
      2029745      0 drwxrwxr-x   3 houeland houeland       60 Sep 19 13:21 /home/houeland/tensorflow_datasets/
      2029746      0 drwxrwxr-x   3 houeland houeland       60 Sep 19 13:21 /home/houeland/tensorflow_datasets/downloads
      2029747      0 drwxrwxr-x   2 houeland houeland      100 Sep 19 13:21 /home/houeland/tensorflow_datasets/downloads/manual
      2029744 6586592 -rw-rw-r--   1 houeland houeland 6744668160 Sep 19 13:20 /home/houeland/tensorflow_datasets/downloads/manual/ILSVRC2012_img_val.tar
      2029740 144422160 -rw-rw-r--   1 houeland houeland 147888291840 Sep 19 11:23 /home/houeland/tensorflow_datasets/downloads/manual/ILSVRC2012_img_train.tar
      2029742  13365020 -rw-rw-r--   1 houeland houeland  13685780480 Sep 19 13:20 /home/houeland/tensorflow_datasets/downloads/manual/ILSVRC2012_img_test.tar
    ```

## Process dataset with TFDS

This will "prepare" the dataset by converting it from the tar and JPEG files above into TFRecord data, which is suitable for loading and using during the training process.

- Install TFDS (other versions should work, but may need slightly different setup)
  - ```
    cd
    virtualenv tfds-prepare
    source tfds-prepare/bin/activate
    pip install tensorflow==2.8.1 tensorflow-datasets==4.4.0
    ```
- Run the "download_and_prepare", which in our case means just processing the already manually downloaded data. (~16 min)
  - ```
    time python3 -c "import tensorflow_datasets as tfds
    tfds.builder('imagenet2012').download_and_prepare()"
    ```
  - ```
    Dataset imagenet2012 downloaded and prepared to /home/houeland/tensorflow_datasets/imagenet2012/5.1.0. Subsequent calls will reuse this data.
    
    real    15m49.703s
    user    11m47.482s
    sys     4m8.679s
    ```
- The data should now be ready to use and we can clean up the rest.
  - ```
    (tfds-prepare) houeland@t1v-n-52e1df7c-w-0:~$ cd ~/tensorflow_datasets
    (tfds-prepare) houeland@t1v-n-52e1df7c-w-0:~/tensorflow_datasets$ du -ch
    156G    ./imagenet2012/5.1.0
    156G    ./imagenet2012
    0       ./downloads/extracted
    157G    ./downloads/manual
    157G    ./downloads
    313G    .
    313G    total
    ```
  - ```
    (tfds-prepare) houeland@t1v-n-52e1df7c-w-0:~/tensorflow_datasets$ rm -r downloads/
    (tfds-prepare) houeland@t1v-n-52e1df7c-w-0:~/tensorflow_datasets$ du -ch
    156G    ./imagenet2012/5.1.0
    156G    ./imagenet2012
    156G    .
    156G    total
    ```

## Optional: Upload dataset to Google Cloud Storage (~3 min)

NOTE: Storing data on GCS is not free, even if access to the TPUs is provided for free through TPU Research Cloud (TRC).

If following this process exactly, the prepared ImageNet data will now just live in-memory on a TPU, which is not ideal since we want to use that memory for training, and because the TPU will restart at some point and then we'd lose everything.

A longer-term place to store these files is in Google Cloud Storage (GCS). TFDS supports reading data out from GCS during training.

- First create a GCS bucket to use for storing files, here I'm using `houeland-trc-ml-datasets-demo`, created within my TRC project.
- ```
  export GCS_TFDS_BUCKET=houeland-trc-ml-datasets-demo
  gsutil -m cp -r ~/tensorflow_datasets gs://${GCS_TFDS_BUCKET}/tensorflow_datasets
  ```

The ImageNet data is now persisted in GCS in a form suitable for use with TFDS. We can now clean up:
- `cd`
- `rm -r tfds-prepare/ /mnt/ramdisk/* tensorflow_datasets`
- ```
  houeland@t1v-n-52e1df7c-w-0:~$ free -g
                total        used        free      shared  buff/cache   available
  Mem:            334           1         329           0           3         330
  Swap:             0           0           0
  ```

## Optional: Train an ImageNet on a TPU using Flax

This assumes that all the previous steps have been completed, so you end up with ImageNet data in GCS that can be read using TFDS.

- Initial setup
  - ```
    export GCS_TFDS_BUCKET=houeland-trc-ml-datasets-demo
    
    virtualenv flax-imagenet
    source flax-imagenet/bin/activate
    pip install "jax[tpu]>=0.2.21" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
- Test that basics are working (we can use TPUs):
  - ```
    (flax-imagenet) houeland@t1v-n-52e1df7c-w-0:~$ python3
    Python 3.8.10 (default, Jun 22 2022, 20:18:18)
    [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import jax
    >>> jax.devices()
    [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]
- Install Flax and train for a few epochs as a quick test (~15 min):
  - ```
    git clone --depth=1 --branch=main https://github.com/google/flax
    cd flax/examples/imagenet/
    pip install -r requirements.txt
    export TFDS_DATA_DIR=gs://${GCS_TFDS_BUCKET}/tensorflow_datasets/
    time python3 main.py --workdir=$HOME/logs/imagenet_tpu_5epochs --config=configs/tpu.py --config.num_epochs=5
    ```
  - If it's working, the output should look something like this:
  - ```
    (flax-imagenet) houeland@t1v-n-52e1df7c-w-0:~/flax/examples/imagenet$ time python3 main.py --workdir=$HOME/logs/imagenet_tpu_5epochs --config=configs/tpu.py --config.num_epochs=5
    2022-09-19 14:14:03.315858: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/lib
    WARNING:absl:GlobalAsyncCheckpointManager is not imported correctly. Checkpointing of GlobalDeviceArrays will not be available.To use the feature, install tensorstore.
    2022-09-19 14:14:04.793035: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/lib
    2022-09-19 14:14:04.793078: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    I0919 14:14:04.811141 140166890789952 xla_bridge.py:345] Unable to initialize backend 'tpu_driver': NOT_FOUND: Unable to find driver in registry given worker:
    I0919 14:14:04.811345 140166890789952 xla_bridge.py:345] Unable to initialize backend 'cuda': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
    I0919 14:14:04.811425 140166890789952 xla_bridge.py:345] Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
    I0919 14:14:08.298967 140166890789952 main.py:50] JAX process: 0 / 1
    I0919 14:14:08.299447 140166890789952 main.py:51] JAX local devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]
    I0919 14:14:08.299590 140166890789952 local.py:45] Setting task status: process_index: 0, process_count: 1
    I0919 14:14:08.299893 140166890789952 local.py:50] Created artifact workdir of type ArtifactType.DIRECTORY and value /home/houeland/logs/imagenet_tpu_5epochs.
    I0919 14:14:08.866074 140166890789952 dataset_info.py:358] Load dataset info from gs://houeland-trc-ml-datasets-demo/tensorflow_datasets/imagenet2012/5.1.0
    I0919 14:14:09.284254 140166890789952 logging_logger.py:35] Constructing tf.data.Dataset imagenet2012 for split train[0:1281167], from gs://houeland-trc-ml-datasets-demo/tensorflow_datasets/imagenet2012/5.1.0
    W0919 14:14:09.457989 140166890789952 options.py:556] options.experimental_threading is deprecated. Use options.threading instead.
    I0919 14:14:09.845413 140166890789952 logging_logger.py:35] Constructing tf.data.Dataset imagenet2012 for split validation[0:50000], from gs://houeland-trc-ml-datasets-demo/tensorflow_datasets/imagenet2012/5.1.0
    W0919 14:14:09.902178 140166890789952 options.py:556] options.experimental_threading is deprecated. Use options.threading instead.
    I0919 14:14:41.801873 140166890789952 checkpoints.py:466] Found no checkpoint files in /home/houeland/logs/imagenet_tpu_5epochs with prefix checkpoint_
    I0919 14:14:41.979685 140166890789952 train.py:331] Initial compilation, this might take some minutes...
    I0919 14:15:08.199315 140166890789952 train.py:337] Initial compilation completed.
    I0919 14:15:19.576108 140166890789952 local.py:50] Created artifact [10] Profile of type ArtifactType.URL and value None.
    I0919 14:15:27.775955 140151934879488 logging_writer.py:35] [100] steps_per_second=2.183623, train_accuracy=0.0013867187080904841, train_learning_rate=0.003165467409417033, train_loss=6.95107364654541
    I0919 14:15:41.729644 140151934879488 logging_writer.py:35] [200] steps_per_second=7.168900, train_accuracy=0.007822265848517418, train_learning_rate=0.009560351260006428, train_loss=6.772485733032227
    I0919 14:15:55.804140 140151934879488 logging_writer.py:35] [300] steps_per_second=7.104685, train_accuracy=0.013154297135770321, train_learning_rate=0.015955235809087753, train_loss=6.586050510406494
    I0919 14:16:10.380145 140151934879488 logging_writer.py:35] [400] steps_per_second=6.861116, train_accuracy=0.016923828050494194, train_learning_rate=0.022350121289491653, train_loss=6.429064750671387
    [...]
    I0919 14:28:25.373429 140151934879488 logging_writer.py:35] [6000] steps_per_second=8.077002, train_accuracy=0.3778710961341858, train_learning_rate=0.3804636299610138, train_loss=2.918736457824707
    I0919 14:28:37.741533 140151934879488 logging_writer.py:35] [6100] steps_per_second=8.084761, train_accuracy=0.3821484446525574, train_learning_rate=0.3868584930896759, train_loss=2.8847763538360596
    I0919 14:28:50.134743 140151934879488 logging_writer.py:35] [6200] steps_per_second=8.069629, train_accuracy=0.3834472596645355, train_learning_rate=0.3932534158229828, train_loss=2.881490468978882
    I0919 14:29:00.699282 140166890789952 train.py:365] eval epoch: 4, loss: 2.9286, accuracy: 37.13
    I0919 14:29:00.699987 140151934879488 logging_writer.py:35] [6255] eval_accuracy=0.3712972104549408, eval_loss=2.928572654724121
    I0919 14:29:00.780086 140166890789952 checkpoints.py:356] Saving checkpoint at step: 6255
    I0919 14:29:01.262479 140166890789952 checkpoints.py:317] Saved checkpoint at /home/houeland/logs/imagenet_tpu_5epochs/checkpoint_6255
    
    real    15m14.530s
    user    653m39.886s
    sys     20m39.698s
    ```
  - If it's NOT working and trying to run on CPU, it might look like this:
  - ```
    ...
    W0919 14:05:00.047981 140045460089920 xla_bridge.py:352] No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
    I0919 14:05:00.048060 140045460089920 main.py:50] JAX process: 0 / 1
    I0919 14:05:00.048120 140045460089920 main.py:51] JAX local devices: [CpuDevice(id=0)]
    ...
    ```
- Start a full training run (~5 hours)
  - `time python3 main.py --workdir=$HOME/logs/imagenet_tpu --config=configs/tpu.py`
  - ```
    ...
    I0919 19:04:52.914754 139814060566272 logging_writer.py:35] [124500] steps_per_second=8.071048, train_accuracy=0.8094823956489563, train_learning_rate=2.9625414754264057e-05, train_loss=0.7803189158439636
    I0919 19:05:05.293230 139814060566272 logging_writer.py:35] [124600] steps_per_second=8.077348, train_accuracy=0.8095800876617432, train_learning_rate=2.1234276573522948e-05, train_loss=0.7774865031242371
    I0919 19:05:17.680801 139814060566272 logging_writer.py:35] [124700] steps_per_second=8.073325, train_accuracy=0.8079199194908142, train_learning_rate=1.423954927304294e-05, train_loss=0.7746614217758179
    I0919 19:05:30.082423 139814060566272 logging_writer.py:35] [124800] steps_per_second=8.063609, train_accuracy=0.8106152415275574, train_learning_rate=8.642912689538207e-06, train_loss=0.7681958675384521
    I0919 19:05:42.479029 139814060566272 logging_writer.py:35] [124900] steps_per_second=8.066568, train_accuracy=0.8128417730331421, train_learning_rate=4.442572389962152e-06, train_loss=0.7664032578468323
    I0919 19:05:54.869171 139814060566272 logging_writer.py:35] [125000] steps_per_second=8.070656, train_accuracy=0.8110741972923279, train_learning_rate=1.6410350553996977e-06, train_loss=0.7732816934585571
    I0919 19:06:07.244970 139814060566272 logging_writer.py:35] [125100] steps_per_second=8.080487, train_accuracy=0.8109472393989563, train_learning_rate=2.3615361044448946e-07, train_loss=0.7680208683013916
    I0919 19:06:11.905154 139829608639552 train.py:365] eval epoch: 99, loss: 0.9427, accuracy: 76.42
    I0919 19:06:11.905771 139814060566272 logging_writer.py:35] [125100] eval_accuracy=0.76422119140625, eval_loss=0.9426663517951965
    I0919 19:06:11.988634 139829608639552 checkpoints.py:356] Saving checkpoint at step: 125100
    I0919 19:06:12.536192 139829608639552 checkpoints.py:317] Saved checkpoint at /home/houeland/logs/imagenet_tpu/checkpoint_125100
    I0919 19:06:12.536425 139829608639552 checkpoints.py:344] Removing checkpoint at /home/houeland/logs/imagenet_tpu/checkpoint_87570
    
    real    266m59.161s
    user    12718m0.591s
    sys     220m13.111s
    ```
  - You can e.g. view progress and metrics using TensorBoard:
    - ```
      virtualenv tensorboard
      pip install tensorboard
      tensorboard --logdir=$HOME/logs
      ```
