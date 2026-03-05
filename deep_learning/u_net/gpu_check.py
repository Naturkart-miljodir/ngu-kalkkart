import os

import tensorflow as tf


def main() -> None:
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"TF_CUDA_PATHS: {os.environ.get('TF_CUDA_PATHS')}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected GPU count: {len(gpus)}")

    if not gpus:
        print("No GPUs detected.")
        return

    for idx, gpu in enumerate(gpus):
        print(f"GPU {idx}: {gpu.name}")


if __name__ == "__main__":
    main()
