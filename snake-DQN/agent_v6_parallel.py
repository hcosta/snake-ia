# agent_v6_parallel.py
# Basado en agent_v6.py, para ser usado con un trainer paralelizado.
# STATE_SIZE es 12.

import os
import random
import numpy as np
from collections import deque

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# TF_CPP_MIN_LOG_LEVEL se establecerá en el trainer_v6_parallel.py

# Condicional para evitar error si TensorFlow no está disponible (aunque es necesario)
try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential, load_model as keras_load_model
except ImportError:
    print("Advertencia: TensorFlow/Keras no encontrado. DQNAgent no funcionará.")
    tf = None
    keras_load_model = None
    Sequential = None
    Dense = None
    Input = None
    Adam = None


AGENT_VERSION = "6_parallel"  # Nueva versión para estos archivos paralelos
DATA_DIR = "trained_data"  # Directorio donde se guardan los modelos


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay_rate=0.9995,
                 epsilon_min=0.01,
                 replay_memory_size=20000, batch_size=64,
                 model_filepath=None,
                 epochs_per_replay=1):

        if tf is None:
            raise ImportError(
                "TensorFlow no está instalado o no se pudo importar. DQNAgent no puede funcionar.")

        self.state_size = state_size  # Debería ser 12 para v6
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if model_filepath is None:
            self.model_filepath = os.path.join(
                DATA_DIR, f'dqn_snake_default_agent_{AGENT_VERSION}.keras')
        else:
            self.model_filepath = model_filepath

        self.epochs_per_replay = epochs_per_replay
        self.memory = deque(maxlen=replay_memory_size)
        self.replay_memory_capacity = replay_memory_size

        model_dir = os.path.dirname(self.model_filepath)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError as e:
                print(
                    f"WARN: No se pudo crear directorio {model_dir} desde DQNAgent: {e}")

        # Lógica de carga/creación del modelo Keras (similar a agent_v6.py)
        if os.path.exists(self.model_filepath):
            print(
                f"INFO: Intentando cargar modelo Keras desde: {self.model_filepath}")
            try:
                self.model = keras_load_model(
                    self.model_filepath, compile=True)
                print(
                    f"INFO: Modelo Keras cargado exitosamente desde {self.model_filepath}")
            except ValueError as ve:
                if "jit_compile" in str(ve) or "is_legacy_optimizer" in str(ve) or "Unable to restore custom metric" in str(ve) or "optimizer" in str(ve).lower():
                    print(
                        f"WARN: Error específico al cargar (posiblemente optimizador/jit_compile): {ve}")
                    print(
                        "INFO: Intentando cargar arquitectura/pesos y recompilando...")
                    try:
                        self.model = keras_load_model(
                            self.model_filepath, compile=False)
                        optimizer = tf.keras.optimizers.Adam(
                            learning_rate=self.learning_rate)
                        self.model.compile(optimizer=optimizer, loss='mse')
                        print(
                            "INFO: Modelo Keras cargado (compile=False) y recompilado.")
                    except Exception as e_recompile:
                        print(
                            f"ERROR: Falló la carga/recompilación: {e_recompile}. Creando nuevo modelo.")
                        self.model = self._build_model()
                else:
                    print(
                        f"WARN: Otro ValueError al cargar modelo: {ve}. Creando nuevo modelo.")
                    self.model = self._build_model()
            except Exception as e:
                print(
                    f"ERROR: Error general al cargar modelo desde '{self.model_filepath}': {e}. Creando nuevo modelo.")
                self.model = self._build_model()
        else:
            print(
                f"INFO: No se encontró modelo en {self.model_filepath}. Creando un nuevo modelo.")
            self.model = self._build_model()

        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Modelo Sequential, igual que en agent_v6.py
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
        state_flat = np.array(state, dtype=np.float32).flatten()
        next_state_flat = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state_flat, action, reward, next_state_flat, done))

    def get_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_np_flat = np.array(state, dtype=np.float32).flatten()
            state_tensor = tf.convert_to_tensor(
                state_np_flat.reshape([1, self.state_size]), dtype=tf.float32)
            act_values_tensor = self.model(state_tensor, training=False)
            return np.argmax(act_values_tensor[0].numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

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

        q_values_current_state_main_model = self.model.predict_on_batch(
            states_np)
        q_values_next_state_target_model = self.target_model.predict_on_batch(
            next_states_np)

        q_targets_batch_np = q_values_current_state_main_model.numpy() if hasattr(
            q_values_current_state_main_model, 'numpy') else np.copy(q_values_current_state_main_model)

        q_values_next_max_np = np.amax(
            q_values_next_state_target_model, axis=1)

        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_max_np * (~dones_np)
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        # Usar tf.data.Dataset para eficiencia (como en agent_v6)
        # Nota: agent_v5_parallel.py usa model.fit(states_np, q_targets_batch_np, ...)
        # agent_v6.py (y agent_v5.py) usa tf.data.Dataset.from_tensor_slices luego .batch()
        # Para consistencia con agent_v6.py, usamos el Dataset approach.
        # Sin embargo, agent_v5_parallel's replay fed the entire batch directly.
        # Let's align with agent_v6 structure which uses .from_tensor_slices -> .batch()
        # Re-evaluating: agent_v5.py / agent_v6.py uses .from_tensors((states_np, q_targets_batch_np))
        # not .from_tensor_slices if we already have the full batch.
        # agent_v5_parallel's self.model.fit(train_dataset, ...) where train_dataset is
        # already batched by prefetch.
        # The original agent_v6.py had:
        # train_dataset = tf.data.Dataset.from_tensor_slices((states_np, q_targets_batch_np))
        # train_dataset = train_dataset.batch(self.batch_size).prefetch(...)
        # This is redundant if states_np *is* the batch.
        # agent_v5.py and agent_v5_parallel.py were more direct:
        # train_dataset = tf.data.Dataset.from_tensors((states_np, q_targets_batch_np))
        # train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)
        # This seems more correct. Let's use this.

        train_dataset = tf.data.Dataset.from_tensors(
            (states_np, q_targets_batch_np))
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)

    def save_keras_model(self):
        model_dir = os.path.dirname(self.model_filepath)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError as e:
                if not os.path.isdir(model_dir):  # Double check
                    print(
                        f"ERROR: Creando dir {model_dir} en save_keras_model: {e}")
                    return
        try:
            self.model.save(self.model_filepath)
            print(f"INFO: Modelo Keras guardado en {self.model_filepath}")
        except Exception as e:
            print(
                f"ERROR: Guardando modelo Keras en {self.model_filepath}: {e}")

    def reset_epsilon_and_memory(self, epsilon_start=None):
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            self.epsilon = self.initial_epsilon
        self.memory.clear()
        print(
            f"INFO: Epsilon reseteado a {self.epsilon:.4f} y memoria limpiada.")
