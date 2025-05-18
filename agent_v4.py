# agent_v4.py

import os
import random
import numpy as np
from collections import deque

# Desactivar warnings y mensajes innecesarios de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 1:
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.layers import Dense  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore

# Descomenta esto si quieres ver dónde se ejecuta cada operación de TF (muy verboso)
# tf.debugging.set_log_device_placement(True)


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,  # Ajustado epsilon_decay
                 replay_memory_size=20000, batch_size=64,
                 model_filepath='dqn_snake_model.weights.h5',  # Cambiado a .weights.h5 como usas
                 epochs_per_replay=1):  # Nuevo parámetro

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon  # Guardar el epsilon inicial para resetear
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_filepath = model_filepath
        self.epochs_per_replay = epochs_per_replay  # Guardar epochs por replay
        self.memory = deque(maxlen=replay_memory_size)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        if os.path.exists(self.model_filepath):
            print(f"Intentando cargar pesos desde {self.model_filepath}...")
            # load_model ya imprime su propio mensaje
            self.load_model(self.model_filepath)
        else:
            print(
                f"No se encontró un modelo guardado en {self.model_filepath}. Iniciando con un nuevo modelo.")

    def _build_model(self):
        model = Sequential()
        # Modelo más grande sugerido
        # Usar input_shape en la primera capa
        model.add(Dense(512, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):  # state aquí es la tupla/lista cruda del entorno
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_np = np.array(state, dtype=np.float32).reshape(
                [1, self.state_size])
            state_tensor = tf.convert_to_tensor(state_np, dtype=tf.float32)
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

        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states_np, dtype=tf.float32)

        q_values_current_state_tf = self.model(states_tf, training=False)
        q_values_next_state_target_model_tf = self.target_model(
            next_states_tf, training=False)

        q_targets_batch_np = q_values_current_state_tf.numpy()
        q_values_next_max_np = np.amax(
            q_values_next_state_target_model_tf.numpy(), axis=1)

        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_max_np * (~dones_np)
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        q_targets_batch_tf = tf.convert_to_tensor(
            q_targets_batch_np, dtype=tf.float32)

        self.model.fit(states_tf, q_targets_batch_tf,
                       epochs=self.epochs_per_replay,  # Usar el nuevo parámetro
                       verbose=0,
                       batch_size=self.batch_size)  # Pasa batch_size a fit también

    def load_model(self, filepath):
        try:
            # Keras >2.6 prefiere .weights.h5 para solo pesos.
            # Si guardaras el modelo completo (model.save()), usarías .keras o un directorio.
            self.model.load_weights(filepath)
            self.update_target_model()
            print(f"Pesos del modelo cargados desde {filepath}")
        except Exception as e:
            print(
                f"Error al cargar pesos desde '{filepath}': {e}. Usando un modelo nuevo.")

    def save_model(self, filepath):
        try:
            self.model.save_weights(filepath)
            print(f"Pesos del modelo guardados en {filepath}")
        except Exception as e:
            print(f"Error al guardar pesos en '{filepath}': {e}")

    def reset_epsilon(self, epsilon_start=None):
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            # Resetea al valor inicial guardado en __init__
            self.epsilon = self.initial_epsilon


if __name__ == '__main__':
    STATE_SIZE_EXAMPLE = 9
    ACTION_SIZE_EXAMPLE = 4
    agent = DQNAgent(STATE_SIZE_EXAMPLE, ACTION_SIZE_EXAMPLE,
                     batch_size=32, epochs_per_replay=5)

    example_state_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
    example_action = agent.get_action(example_state_tuple)
    example_reward = 10.0  # Usar float para recompensas
    example_next_state_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
    example_done = False

    agent.remember(example_state_tuple, example_action,
                   example_reward, example_next_state_tuple, example_done)
    print(f"Memoria después de una transición: {len(agent.memory)}")

    for _ in range(agent.batch_size * 2):
        s_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
        a = agent.get_action(s_tuple)
        r = np.random.randint(-10, 11).astype(np.float32)  # Recompensas float
        s_next_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
        d = random.choice([True, False])
        agent.remember(s_tuple, a, r, s_next_tuple, d)

    print(f"Memoria antes de replay: {len(agent.memory)}")
    if len(agent.memory) >= agent.batch_size:
        agent.replay()
        print("Replay ejecutado.")

    agent.update_target_model()
    print("Modelo objetivo actualizado.")

    test_model_path = "test_dqn_agent_v4.weights.h5"
    agent.save_model(test_model_path)
    agent.load_model(test_model_path)
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    print("Prueba de agente v4 finalizada.")
