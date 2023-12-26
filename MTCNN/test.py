import tensorflow as tf

# Check if TensorFlow is built with CUDA (GPU)
print("Built with CUDA: ", tf.test.is_built_with_cuda())

# Check if TensorFlow can access a GPU
print("GPU Available: ", tf.test.is_gpu_available())