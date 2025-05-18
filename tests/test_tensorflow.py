import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 1:
    import tensorflow as tf

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"Num GPUs Available: {len(gpus)}")

if gpus:
    try:
        for gpu in gpus:
            print(f"Name: {gpu.name}, Type: {gpu.device_type}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print("Matriz resultante calculada en GPU:")
        print(c.numpy())
        print("¡TensorFlow está usando la GPU!")
    except RuntimeError as e:
        print(f"Error durante la prueba de GPU: {e}")
else:
    print("TensorFlow NO detecta ninguna GPU.")

# docker run --gpus all -it -d --name tf_2_19_gpu -v "C:\Users\hcost\Desktop\Snake IA":/app tensorflow/tensorflow:2.19.0-gpu sleep infinity

# docker run --gpus all -it --rm ` -v "C:\Users\hcost\Desktop\Snake IA:/app" ` --workdir "/app" ` --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 ` --name "tf_gpu_container" ` nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# xvfb-run -a python3 trainer_v3.py --no-visualize <-- entrenamiento sin salida gráfica

# xvfb-run -a python3 -m cProfile -o training.prof trainer_v3.py --no-visualize --episodes 10

# xvfb-run trainer_v3.py --no-visualize --episodes 10
