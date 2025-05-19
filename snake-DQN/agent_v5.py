# agent_v5.py

"""
Correcciones de la versión 5
============================
* Se ha modificado los scripts agent_v4.py y trainer_v4.py para usar el método recomendado de Keras para guardar y cargar modelos completos (model.save() y tf.keras.models.load_model())
* Se ha implementado numba y compilación jit para optimizar la lógica del juego y el estado del agente.
* Además se guarda el estado del entrenamiento para continuar desde el último episodio excepto si se hace un reset.
* La velocidad de entrenamiento ha mejorado más de 10 veces en comparación con la versión 4.
"""

# agent_v5.py
# Con soporte para que el trainer maneje el checkpointing y los paths.

import os
import random
import numpy as np
from collections import deque
# pickle no es necesario aquí si el trainer maneja el pkl del estado completo

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if 1:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential, load_model as keras_load_model

AGENT_VERSION = "5"
DATA_DIR = "trained_data"


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay_rate=0.9995,
                 epsilon_min=0.01,
                 replay_memory_size=20000, batch_size=64,
                 # Default con subdir
                 model_filepath=f'{DATA_DIR}/dqn_snake_default_agent_{AGENT_VERSION}.keras',
                 epochs_per_replay=1):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # ESTE self.model_filepath ES EL IMPORTANTE. El trainer.py le pasará el path completo
        # que ya incluye "trained_data/", por ejemplo: "trained_data/dqn_snake_checkpoint.keras"
        self.model_filepath = model_filepath
        self.epochs_per_replay = epochs_per_replay
        self.memory = deque(maxlen=replay_memory_size)
        self.replay_memory_capacity = replay_memory_size

        # (Opcional) Crear directorio para el modelo si no existe.
        # Es mejor que el trainer lo haga, pero esto es una salvaguarda.
        model_dir = os.path.dirname(self.model_filepath)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir, exist_ok=True)
                # print(f"Directorio para modelo Keras creado por agente (si no existía): {model_dir}")
            except OSError as e:
                print(
                    f"Advertencia: No se pudo crear directorio {model_dir} desde agente: {e}")

        # Lógica de carga del modelo Keras, incluyendo el workaround para jit_compile
        if os.path.exists(self.model_filepath):
            try:
                print(
                    f"Intentando cargar modelo Keras desde {self.model_filepath}...")
                self.model = keras_load_model(self.model_filepath)
                print(
                    f"Modelo Keras cargado exitosamente desde {self.model_filepath}")
            except ValueError as ve:
                if "Argument(s) not recognized" in str(ve) and \
                   ("jit_compile" in str(ve) or "is_legacy_optimizer" in str(ve)):
                    print(
                        f"Error específico de optimizador al cargar (jit_compile/is_legacy): {ve}")
                    print("Intentando cargar arquitectura/pesos y recompilando...")
                    try:
                        self.model = keras_load_model(
                            self.model_filepath, compile=False)
                        # Usar self.learning_rate aquí es importante para que el optimizador
                        # se cree con el LR esperado, especialmente si este agente
                        # fuera a continuar entrenando (aunque para play-only no es crítico).
                        optimizer = tf.keras.optimizers.Adam(
                            learning_rate=self.learning_rate)
                        # O la función de pérdida que uses
                        self.model.compile(optimizer=optimizer, loss='mse')
                        print(
                            "Modelo Keras cargado con arquitectura/pesos y recompilado.")
                    except Exception as e_recompile:
                        print(
                            f"FALLO CRÍTICO al cargar/recompilar arquitectura y pesos: {e_recompile}")
                        print(
                            "Creando un nuevo modelo Keras desde cero como último recurso.")
                        self.model = self._build_model()  # Fallback final
                else:
                    # Otro ValueError no relacionado
                    print(
                        f"Otro ValueError al cargar modelo Keras: {ve}. Creando nuevo modelo.")
                    self.model = self._build_model()  # Fallback
            except Exception as e:  # Otras excepciones generales
                print(
                    f"Error general al cargar modelo Keras desde '{self.model_filepath}': {e}.")
                print("Creando un nuevo modelo Keras desde cero.")
                self.model = self._build_model()  # Fallback
        else:
            print(
                f"No se encontró modelo Keras en {self.model_filepath}. Creando un nuevo modelo.")
            self.model = self._build_model()

        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
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

        q_values_current_state = self.model.predict_on_batch(states_np)
        q_values_next_state_target_model = self.target_model.predict_on_batch(
            next_states_np)

        q_targets_batch_np = q_values_current_state.numpy() if hasattr(
            q_values_current_state, 'numpy') else np.copy(q_values_current_state)
        q_next_state_eval_np = q_values_next_state_target_model.numpy() if hasattr(
            q_values_next_state_target_model, 'numpy') else np.copy(q_values_next_state_target_model)

        q_values_next_max_np = np.amax(q_next_state_eval_np, axis=1)

        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_max_np * (~dones_np)
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        train_dataset = tf.data.Dataset.from_tensors(
            (states_np, q_targets_batch_np))
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.model.fit(train_dataset, epochs=self.epochs_per_replay, verbose=0)

    def save_keras_model(self):
        """Guarda solo el modelo Keras. El path (self.model_filepath) ya incluye el subdirectorio."""
        # Asegurar que el directorio existe (aunque el trainer debería haberlo hecho)
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
        """Resetea epsilon a su valor inicial y limpia la memoria."""
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            self.epsilon = self.initial_epsilon
        self.memory.clear()
        print(f"Epsilon reseteado a {self.epsilon:.4f} y memoria limpiada.")
