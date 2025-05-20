# agent_v6.py

"""
Versión 6
=========
* Introduce una nueva variable de estado: Accesibilidad de la Cola Después de Comer.
* El state_size se espera que sea 12 (manejado por el trainer al instanciar).
* AGENT_VERSION actualizada a "6".
"""

import os
import random
import numpy as np
from collections import deque

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suprime logs informativos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Condicional para evitar error si TensorFlow no está disponible (aunque es necesario)
try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
except ImportError:
    print("Advertencia: TensorFlow/Keras no encontrado. DQNAgent no funcionará.")
    tf = None  # Placeholder
    keras_load_model = None
    Sequential = None
    Dense = None
    Input = None
    Adam = None


AGENT_VERSION = "6"  # Actualizado para v6
DATA_DIR = "trained_data"  # Directorio donde se guardan los modelos


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay_rate=0.9995,
                 epsilon_min=0.01,
                 replay_memory_size=20000, batch_size=64,
                 model_filepath=None,  # El trainer debe pasar el path completo
                 epochs_per_replay=1):

        if tf is None:  # Comprobar si TensorFlow está disponible
            raise ImportError(
                "TensorFlow no está instalado o no se pudo importar. DQNAgent no puede funcionar.")

        self.state_size = state_size  # Ahora se espera 12
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon  # Guardar el epsilon inicial para reset
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if model_filepath is None:
            # Path por defecto si no se proporciona (el trainer usualmente lo hará)
            self.model_filepath = os.path.join(
                DATA_DIR, f'dqn_snake_default_agent_{AGENT_VERSION}.keras')
        else:
            self.model_filepath = model_filepath

        self.epochs_per_replay = epochs_per_replay
        self.memory = deque(maxlen=replay_memory_size)
        # Para referencia en trainer si es necesario
        self.replay_memory_capacity = replay_memory_size

        model_dir = os.path.dirname(self.model_filepath)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError as e:
                print(
                    f"Advertencia: No se pudo crear directorio {model_dir} desde agente: {e}")

        # Lógica de carga/creación del modelo Keras
        if os.path.exists(self.model_filepath):
            try:
                print(
                    f"Intentando cargar modelo Keras desde {self.model_filepath}...")
                # Usar compile=True por defecto si es posible, o manejar errores específicos.
                self.model = keras_load_model(
                    self.model_filepath, compile=True)
                print(
                    f"Modelo Keras cargado exitosamente desde {self.model_filepath}")
            except ValueError as ve:  # Errores comunes de incompatibilidad de optimizador
                if "jit_compile" in str(ve) or "is_legacy_optimizer" in str(ve) or "Unable to restore custom metric" in str(ve) or "optimizer" in str(ve).lower():
                    print(
                        f"Error específico al cargar (posiblemente optimizador/jit_compile): {ve}")
                    print("Intentando cargar arquitectura/pesos y recompilando...")
                    try:
                        self.model = keras_load_model(
                            self.model_filepath, compile=False)
                        optimizer = tf.keras.optimizers.Adam(
                            learning_rate=self.learning_rate)
                        # Asume 'mse' como pérdida
                        self.model.compile(optimizer=optimizer, loss='mse')
                        print(
                            "Modelo Keras cargado con arquitectura/pesos y recompilado.")
                    except Exception as e_recompile:
                        print(
                            f"FALLO CRÍTICO al cargar/recompilar: {e_recompile}")
                        print(
                            "Creando un nuevo modelo Keras desde cero como último recurso.")
                        self.model = self._build_model()
                else:
                    print(
                        f"Otro ValueError al cargar modelo Keras: {ve}. Creando nuevo modelo.")
                    self.model = self._build_model()
            except Exception as e:
                print(
                    f"Error general al cargar modelo Keras ('{self.model_filepath}'): {e}. Creando nuevo modelo.")
                self.model = self._build_model()
        else:
            print(
                f"No se encontró modelo Keras en {self.model_filepath}. Creando un nuevo modelo.")
            self.model = self._build_model()

        # Siempre construir un nuevo target model
        self.target_model = self._build_model()
        self.update_target_model()  # Sincronizar pesos iniciales

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),  # state_size será 12
            Dense(512, activation='relu', dtype='float32'),
            Dense(512, activation='relu', dtype='float32'),
            Dense(256, activation='relu', dtype='float32'),
            Dense(self.action_size, activation='linear', dtype='float32')
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def update_target_model(self):
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Asegurar que el estado se aplane y sea float32 para la memoria
        state_flat = np.array(state, dtype=np.float32).flatten()
        next_state_flat = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state_flat, action, reward, next_state_flat, done))

    def get_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Aplanar y convertir estado para predicción
            state_np_flat = np.array(state, dtype=np.float32).flatten()
            state_tensor = tf.convert_to_tensor(
                state_np_flat.reshape([1, self.state_size]), dtype=tf.float32)
            act_values_tensor = self.model(
                state_tensor, training=False)  # Usar el modelo principal
            return np.argmax(act_values_tensor[0].numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # No hay suficientes muestras para un batch

        minibatch = random.sample(self.memory, self.batch_size)

        # Desempaquetar y convertir a arrays NumPy
        states_np = np.array([experience[0]
                             for experience in minibatch], dtype=np.float32)
        actions_np = np.array([experience[1]
                              for experience in minibatch], dtype=np.int32)
        rewards_np = np.array([experience[2]
                              for experience in minibatch], dtype=np.float32)
        next_states_np = np.array([experience[3]
                                  for experience in minibatch], dtype=np.float32)
        dones_np = np.array([experience[4]
                            for experience in minibatch], dtype=bool)

        # Predicciones
        # Q-values para el estado actual desde el modelo principal (para actualizar)
        q_values_current_state_main_model = self.model.predict_on_batch(
            states_np)

        # Q-values para el siguiente estado desde el modelo objetivo (para calcular el target Q)
        q_values_next_state_target_model = self.target_model.predict_on_batch(
            next_states_np)

        # Copiar q_values_current_state para modificar solo las acciones tomadas
        q_targets_batch_np = q_values_current_state_main_model.numpy() if hasattr(
            q_values_current_state_main_model, 'numpy') else np.copy(q_values_current_state_main_model)

        # Máximo Q-value para el siguiente estado (de target_model)
        q_values_next_max_np = np.amax(
            q_values_next_state_target_model, axis=1)

        # Cálculo del Q-target: R + gamma * max_a' Q_target(s', a')
        # Si done es True, el target es solo R
        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_max_np * (~dones_np)

        # Asegurar que para los estados terminales, el target es solo la recompensa
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        # Actualizar los Q-values para las acciones que se tomaron
        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        # Entrenar el modelo principal con los targets calculados
        # Usar tf.data.Dataset para eficiencia
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (states_np, q_targets_batch_np))
        train_dataset = train_dataset.batch(
            self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)

    def save_keras_model(self):
        """Guarda el modelo Keras. El path (self.model_filepath) debe ser completo."""
        model_dir = os.path.dirname(self.model_filepath)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError as e:
                # Doble chequeo por si se creó entretanto
                if not os.path.isdir(model_dir):
                    print(
                        f"Error al crear directorio {model_dir} en save_keras_model: {e}")
                    return  # No intentar guardar si el directorio no se puede crear
        try:
            self.model.save(self.model_filepath)
            print(f"Modelo Keras guardado en {self.model_filepath}")
        except Exception as e:
            print(
                f"Error al guardar modelo Keras en {self.model_filepath}: {e}")

    def reset_epsilon_and_memory(self, epsilon_start=None):
        """Resetea epsilon a su valor inicial (o uno dado) y limpia la memoria."""
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            self.epsilon = self.initial_epsilon  # Usar el epsilon inicial guardado
        self.memory.clear()
        print(f"Epsilon reseteado a {self.epsilon:.4f} y memoria limpiada.")
