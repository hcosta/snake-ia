# agent_v10.py

"""
Dueling Double DQN (Dueling DDQN)
"""
import os
import random
import numpy as np
from collections import deque

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Input, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model as keras_load_model
except ImportError:
    # ... (imports nulos como antes) ...
    print("Advertencia: TensorFlow/Keras no encontrado. DQNAgent no funcionará.")
    tf = None
    keras_load_model = None
    # Sequential ya no está aquí
    Dense = None
    Input = None
    Lambda = None  # Añadido aquí también
    Model = None  # Añadido aquí también
    Adam = None


# --- CAMBIO IMPORTANTE: Mover combine_streams a nivel de módulo ---
# Keras necesita poder encontrar esta función por su nombre al cargar el modelo.
def combine_streams(streams):  # El nombre debe coincidir con el que se usó al guardar
    v_s, a_s = streams
    return v_s + (a_s - tf.reduce_mean(a_s, axis=1, keepdims=True))


# Mantener o incrementar si se considera una nueva subversión con el fix
AGENT_VERSION = "10_v4_DuelingDDQN"
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

        self.state_size = state_size
        # ... (resto de la inicialización de atributos como en tu agent_v10.py) ...
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
                # --- CAMBIO IMPORTANTE: Añadir 'combine_streams' a custom_objects ---
                custom_objects = {'combine_streams': combine_streams}
                self.model = keras_load_model(
                    self.model_filepath,
                    compile=True,  # Intentar compilar directamente
                    custom_objects=custom_objects
                )
                print(
                    f"Modelo Keras cargado exitosamente desde {self.model_filepath}")
            except Exception as e:  # Captura más genérica inicialmente para el fallback
                print(
                    f"Error al cargar modelo compilado o con estructura específica: {e}")
                print("Intentando cargar solo la arquitectura y recompilar...")
                try:
                    custom_objects = {'combine_streams': combine_streams}
                    self.model = keras_load_model(
                        self.model_filepath,
                        compile=False,  # No compilar aquí para recompilar manualmente
                        custom_objects=custom_objects
                    )
                    # Recompilar el modelo después de cargarlo con compile=False
                    optimizer_load = tf.keras.optimizers.Adam(
                        learning_rate=self.learning_rate)
                    self.model.compile(optimizer=optimizer_load, loss='mse')
                    print(
                        "Modelo Keras cargado con arquitectura y recompilado exitosamente.")
                except Exception as e_recompile:
                    print(
                        f"FALLO CRÍTICO al cargar/recompilar: {e_recompile}")
                    print(
                        "Creando un nuevo modelo Keras (Dueling) desde cero.")
                    self.model = self._build_model()
        else:
            print(
                f"No se encontró modelo Keras en {self.model_filepath}. Creando un nuevo modelo (Dueling).")
            self.model = self._build_model()

        self.target_model = self._build_model()  # Esto también usará la función global
        self.update_target_model()

    def _build_model(self):
        input_layer = Input(shape=(self.state_size,))
        base_layer = Dense(64, activation='relu')(input_layer)

        value_stream = Dense(32, activation='relu')(base_layer)
        value = Dense(1, activation='linear',
                      name='value_output')(value_stream)

        advantage_stream = Dense(32, activation='relu')(base_layer)
        advantages = Dense(self.action_size, activation='linear',
                           name='advantage_output')(advantage_stream)

        # --- CAMBIO IMPORTANTE: Usar la función 'combine_streams' definida a nivel de módulo ---
        # La capa Lambda ahora hace referencia a la función global.
        output_layer = Lambda(combine_streams, name='q_values_output')(
            [value, advantages])

        model = Model(inputs=input_layer, outputs=output_layer)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        # model.summary()
        return model

    # ... El resto de los métodos (update_target_model, remember, get_action, replay, etc.)
    # permanecen iguales que en tu agent_v10.py (Dueling DDQN) que ya tenías.
    # Solo he copiado la estructura de esos métodos de tu agent_v9.py para completitud.

    def update_target_model(self):
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        state_flat = np.array(state, dtype=np.float32).flatten()
        next_state_flat = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state_flat, action, reward, next_state_flat, done))

    def get_action(self, state):
        state_np = np.array(state, dtype=np.float32)
        if random.random() <= self.epsilon:
            safe_actions = []
            # Asumiendo que state_np[0:3] son danger_straight, danger_left, danger_right
            # Esto puede necesitar ajuste si el orden de tu estado de 15 elementos es diferente
            # para estas características específicas. Aquí se asume que las primeras 3 son los peligros.
            if self.state_size >= 3:  # Chequeo de seguridad
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

        q_values_current_state_main_model_tensor = self.model.predict_on_batch(
            states_np)
        q_values_next_state_main_model_tensor = self.model.predict_on_batch(
            next_states_np)
        q_values_next_state_target_model_tensor = self.target_model.predict_on_batch(
            next_states_np)

        q_values_current_state_main_model_np = q_values_current_state_main_model_tensor.numpy() \
            if hasattr(q_values_current_state_main_model_tensor, 'numpy') else q_values_current_state_main_model_tensor
        q_values_next_state_main_model_np = q_values_next_state_main_model_tensor.numpy() \
            if hasattr(q_values_next_state_main_model_tensor, 'numpy') else q_values_next_state_main_model_tensor
        q_values_next_state_target_model_np = q_values_next_state_target_model_tensor.numpy() \
            if hasattr(q_values_next_state_target_model_tensor, 'numpy') else q_values_next_state_target_model_tensor

        q_targets_batch_np = np.copy(q_values_current_state_main_model_np)
        best_actions_next_state_indices = np.argmax(
            q_values_next_state_main_model_np, axis=1)
        q_values_next_state_ddqn = q_values_next_state_target_model_np[
            np.arange(self.batch_size), best_actions_next_state_indices
        ]

        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_state_ddqn * (~dones_np)
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (states_np, q_targets_batch_np))
        train_dataset = train_dataset.batch(
            self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)

    def save_keras_model(self):
        # ... (como en tu agent_v10.py, asegurando que el directorio existe) ...
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
