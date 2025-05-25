# agent_v9.py

"""
DDQN
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
AGENT_VERSION = "9_DDQN"
DATA_DIR = "trained_data"


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005, discount_factor=0.9,
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
            except ValueError as ve:  # Manejo de errores como en agent_v8.py
                if "jit_compile" in str(ve) or "is_legacy_optimizer" in str(ve) or "Unable to restore custom metric" in str(ve) or "optimizer" in str(ve).lower():
                    print(
                        f"Error específico al cargar (posiblemente optimizador/jit_compile): {ve}")
                    print("Intentando cargar arquitectura/pesos y recompilando...")
                    try:
                        self.model = keras_load_model(
                            self.model_filepath, compile=False)
                        optimizer_load = tf.keras.optimizers.Adam(
                            learning_rate=self.learning_rate)
                        self.model.compile(
                            optimizer=optimizer_load, loss='mse')
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

    def _build_model(self):  # Sin cambios respecto a agent_v8.py
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def update_target_model(self):  # Sin cambios
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):  # Sin cambios
        state_flat = np.array(state, dtype=np.float32).flatten()
        next_state_flat = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state_flat, action, reward, next_state_flat, done))

    # Sin cambios respecto a agent_v8.py (a menos que quieras quitar el zigzag)
    def get_action(self, state):
        state_np = np.array(state, dtype=np.float32)
        if random.random() <= self.epsilon:
            safe_actions = []
            danger_straight, danger_left, danger_right = state_np[0], state_np[1], state_np[2]
            if not danger_straight:
                safe_actions.append(0)
            if not danger_left:
                safe_actions.append(1)
            if not danger_right:
                safe_actions.append(2)
            if safe_actions:
                return random.choice(safe_actions)
            return np.argmax(self.model.predict(state_np.reshape(1, -1), verbose=0)[0])
        else:
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

        # --- INICIO DE CAMBIOS PARA DOUBLE DQN ---

        # 1. Predecir Q-values para S_t con el modelo principal (online network)
        #    Esto se usa para construir el target final, ya que q_targets_batch_np se basa en esto.
        q_values_current_state_main_model_tensor = self.model.predict_on_batch(
            states_np)

        # 2. Predecir Q-values para S_{t+1} con el modelo principal (online_network)
        #    para SELECCIONAR la mejor acción a_*.
        q_values_next_state_main_model_tensor = self.model.predict_on_batch(
            next_states_np)

        # 3. Predecir Q-values para S_{t+1} con el modelo objetivo (target_network)
        #    para EVALUAR la acción a_* seleccionada por el modelo principal.
        q_values_next_state_target_model_tensor = self.target_model.predict_on_batch(
            next_states_np)

        # Convertir a NumPy arrays si son tensores
        q_values_current_state_main_model_np = q_values_current_state_main_model_tensor.numpy() \
            if hasattr(q_values_current_state_main_model_tensor, 'numpy') else q_values_current_state_main_model_tensor

        q_values_next_state_main_model_np = q_values_next_state_main_model_tensor.numpy() \
            if hasattr(q_values_next_state_main_model_tensor, 'numpy') else q_values_next_state_main_model_tensor

        q_values_next_state_target_model_np = q_values_next_state_target_model_tensor.numpy() \
            if hasattr(q_values_next_state_target_model_tensor, 'numpy') else q_values_next_state_target_model_tensor

        # Copiar q_values_current_state para modificar solo las acciones tomadas
        q_targets_batch_np = np.copy(q_values_current_state_main_model_np)

        # Para Double DQN:
        # a. Seleccionar las mejores acciones para el siguiente estado (S_{t+1}) usando el MODELO PRINCIPAL (online)
        best_actions_next_state_indices = np.argmax(
            q_values_next_state_main_model_np, axis=1)

        # b. Obtener los Q-values de esas mejores acciones, pero del MODELO OBJETIVO (target)
        #    Usamos np.arange(self.batch_size) para indexar correctamente cada muestra del batch.
        q_values_next_state_ddqn = q_values_next_state_target_model_np[
            np.arange(self.batch_size), best_actions_next_state_indices
        ]

        # Cálculo del Q-target para Double DQN: R + gamma * Q_target(S', argmax_a' Q_online(S',a'))
        # Si done es True, el target es solo R
        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_state_ddqn * (~dones_np)

        # --- FIN DE CAMBIOS PARA DOUBLE DQN ---

        # Asegurar que para los estados terminales, el target es solo la recompensa
        # (Esta línea es importante y se mantiene igual)
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        # Actualizar los Q-values en q_targets_batch_np para las acciones que realmente se tomaron
        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        # Entrenar el modelo principal con los targets calculados
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (states_np, q_targets_batch_np))
        train_dataset = train_dataset.batch(
            self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)

    def save_keras_model(self):  # Sin cambios
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

    def reset_epsilon_and_memory(self, epsilon_start=None):  # Sin cambios
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            self.epsilon = self.initial_epsilon
        self.memory.clear()
        print(f"Epsilon reseteado a {self.epsilon:.4f} y memoria limpiada.")
