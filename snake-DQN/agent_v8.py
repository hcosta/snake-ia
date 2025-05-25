# agent_v7.py

"""
Versión 7
=========
* AGENT_VERSION actualizada a "7".
* Arquitectura del modelo (_build_model) simplificada a una capa oculta de 256 neuronas,
  según el informe de referencia.
* state_size y action_size se esperan que sean 11 y 3 respectivamente,
  configurados por el trainer.
"""

import os
import random
import numpy as np
from collections import deque

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# CAMBIO: Versión del agente actualizada.
AGENT_VERSION = "8"
DATA_DIR = "trained_data"


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005, discount_factor=0.9,
                 # Este decay rate será reemplazado por lineal en el trainer
                 epsilon=1.0, epsilon_decay_rate=0.9995,
                 epsilon_min=0.01,
                 replay_memory_size=20000, batch_size=64,
                 model_filepath=None,
                 epochs_per_replay=1):

        if tf is None:
            raise ImportError(
                "TensorFlow no está instalado o no se pudo importar. DQNAgent no puede funcionar.")

        # CAMBIO: state_size se espera que sea 11, action_size se espera que sea 3.
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        # Mantenido por si se usa, pero el trainer implementará lineal
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
                    f"Advertencia: No se pudo crear directorio {model_dir} desde agente: {e}")

        if os.path.exists(self.model_filepath):
            try:
                print(
                    f"Intentando cargar modelo Keras desde {self.model_filepath}...")
                self.model = keras_load_model(
                    self.model_filepath, compile=True)
                print(
                    f"Modelo Keras cargado exitosamente desde {self.model_filepath}")
            except ValueError as ve:
                if "jit_compile" in str(ve) or "is_legacy_optimizer" in str(ve) or "Unable to restore custom metric" in str(ve) or "optimizer" in str(ve).lower():
                    print(
                        f"Error específico al cargar (posiblemente optimizador/jit_compile): {ve}")
                    print("Intentando cargar arquitectura/pesos y recompilando...")
                    try:
                        self.model = keras_load_model(
                            self.model_filepath, compile=False)
                        optimizer = tf.keras.optimizers.Adam(  # AdamW es una buena alternativa también
                            learning_rate=self.learning_rate)
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

        self.target_model = self._build_model()
        self.update_target_model()

    # agent_v7.py (volver a una arquitectura más simple)
    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),  # Menos neuronas
            Dense(self.action_size, activation='linear')
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
        # Convert the state tuple to a NumPy array
        state_np = np.array(state, dtype=np.float32)

        if random.random() <= self.epsilon:
            # Solo permite acciones que no lleven a colisión inmediata
            safe_actions = []
            # state_np can be used here directly as it's an array
            danger_straight, danger_left, danger_right = state_np[0], state_np[1], state_np[2]
            if not danger_straight:
                safe_actions.append(0)  # Recto es seguro
            if not danger_left:
                safe_actions.append(1)   # Izquierda rel. es segura
            if not danger_right:
                safe_actions.append(2)   # Derecha rel. es segura

            if safe_actions:
                return random.choice(safe_actions)
            else:
                # Use the converted NumPy array here
                return np.argmax(self.model.predict(state_np.reshape(1, -1), verbose=0)[0])
        else:
            # Acción greedy
            # Use the converted NumPy array here
            return np.argmax(self.model.predict(state_np.reshape(1, -1), verbose=0)[0])

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

        # Predicciones
        q_values_current_state_main_model_tensor = self.model.predict_on_batch(
            states_np)
        q_values_next_state_target_model_tensor = self.target_model.predict_on_batch(
            next_states_np)

        # Convertir a NumPy arrays si son tensores
        q_values_current_state_main_model_np = q_values_current_state_main_model_tensor.numpy() \
            if hasattr(q_values_current_state_main_model_tensor, 'numpy') else q_values_current_state_main_model_tensor
        q_values_next_state_target_model_np = q_values_next_state_target_model_tensor.numpy() \
            if hasattr(q_values_next_state_target_model_tensor, 'numpy') else q_values_next_state_target_model_tensor

        # Copiar q_values_current_state para modificar solo las acciones tomadas
        q_targets_batch_np = np.copy(q_values_current_state_main_model_np)

        # Máximo Q-value para el siguiente estado (de target_model)
        q_values_next_max_np = np.amax(
            q_values_next_state_target_model_np, axis=1)

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
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (states_np, q_targets_batch_np))
        train_dataset = train_dataset.batch(
            self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)

    def save_keras_model(self):
        model_dir = os.path.dirname(self.model_filepath)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError as e:
                if not os.path.isdir(model_dir):
                    print(
                        f"Error al crear directorio {model_dir} en save_keras_model: {e}")
                    return
        try:
            self.model.save(self.model_filepath)
            print(f"Modelo Keras guardado en {self.model_filepath}")
        except Exception as e:
            print(
                f"Error al guardar modelo Keras en {self.model_filepath}: {e}")

    def reset_epsilon_and_memory(self, epsilon_start=None):
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            self.epsilon = self.initial_epsilon
        self.memory.clear()
        print(f"Epsilon reseteado a {self.epsilon:.4f} y memoria limpiada.")
