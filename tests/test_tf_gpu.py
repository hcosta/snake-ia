import os
import tensorflow as tf
import unittest

# Desactivar algunas optimizaciones de oneDNN y logs de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestTensorFlowGPU(unittest.TestCase):
    def test_tensorflow_gpu_available(self):
        print(f"TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Num GPUs Available: {len(gpus)}")

        if not gpus:
            self.fail("TensorFlow NO detecta ninguna GPU.")
        else:
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
                print("¡TensorFlow está usando la GPU correctamente!")
            except RuntimeError as e:
                self.fail(f"Error durante la prueba de GPU: {e}")


if __name__ == '__main__':
    unittest.main()
