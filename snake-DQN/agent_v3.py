# agent_v3.py: Optimizado para tensorflow 2.19.0 y GPU
# Principales Cambios Propuestos:
#   - Consistencia en el Tipo de Dato (np.float32): Asegurar que los estados se manejen como np.float32 desde que se guardan en la memoria de repetición.
#   - Conversión a Tensores y Llamada Directa al Modelo: En get_action y replay, convertir los arrays de NumPy a tf.Tensor de tipo tf.float32 una vez y luego llamar al modelo directamente (ej: self.model(tensor_input, training=False)) en lugar de usar self.model.predict(). Esto puede ser más eficiente y reducir el overhead de predict.
#   - Vectorización del Cálculo de Q-Targets en replay(): Reemplazar el bucle Python explícito para calcular los Q-targets con operaciones vectorizadas de NumPy. Esto es un cambio más grande pero puede dar una mejora sustancial en la parte de CPU de la preparación del batch.
#   - Log de Ubicación de Dispositivos (Opcional para esta versión, pero útil para depurar): Mantener o añadir tf.debugging.set_log_device_placement(True) al inicio del script si aún quieres verificar la ubicación de las operaciones.

import random
import numpy as np
from collections import deque  # Para la memoria de repetición
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import os

# Descomenta esto si quieres ver dónde se ejecuta cada operación de TF (muy verboso)
# tf.debugging.set_log_device_placement(True)


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 replay_memory_size=10000, batch_size=64, model_filepath='dqn_snake_model.keras'):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_filepath = model_filepath
        self.memory = deque(maxlen=replay_memory_size)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        if os.path.exists(self.model_filepath):
            self.load_model(self.model_filepath)
            print(f"Modelo cargado desde {self.model_filepath}")
        else:
            print(
                f"No se encontró un modelo guardado en {self.model_filepath}. Iniciando con un nuevo modelo.")

    def _build_model(self):
        model = Sequential()
        # Modelo actual (pequeño)
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))

        # Modelo más grande que sugerí para probar carga en GPU (puedes elegir cuál usar)
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Almacena una transición (experiencia) en la memoria de repetición."""
        # CAMBIO: Asegurar que el estado se guarda como np.float32 y aplanado
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """Elige una acción usando la política epsilon-greedy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # CAMBIO: Preparar el estado como tensor float32 y llamar al modelo directamente
            state_np = np.array(state, dtype=np.float32).reshape(
                [1, self.state_size])
            state_tensor = tf.convert_to_tensor(state_np, dtype=tf.float32)
            # Usar training=False para inferencia
            act_values_tensor = self.model(state_tensor, training=False)
            # Convertir a NumPy para argmax
            return np.argmax(act_values_tensor[0].numpy())

    def replay(self):
        """Entrena la red neuronal usando un lote de experiencias de la memoria."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # CAMBIO: Extraer y preparar arrays de NumPy con dtype=np.float32
        # Si 'remember' ya los guarda como np.float32.flatten(), esto es más directo
        states_np = np.array([experience[0]
                             for experience in minibatch], dtype=np.float32)
        # Las acciones son índices
        actions_np = np.array([experience[1]
                              for experience in minibatch], dtype=np.int32)
        rewards_np = np.array([experience[2]
                              for experience in minibatch], dtype=np.float32)
        next_states_np = np.array([experience[3]
                                  for experience in minibatch], dtype=np.float32)
        dones_np = np.array([experience[4]
                            for experience in minibatch], dtype=bool)

        # CAMBIO: Convertir a Tensores una vez y llamar a los modelos directamente
        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states_np, dtype=tf.float32)

        # Obtener Q-values actuales y futuros (del target model) como tensores
        # Usar training=False para que las capas como Dropout o BatchNormalization (si las tuvieras)
        # se comporten en modo inferencia.
        q_values_current_state_tf = self.model(states_tf, training=False)
        q_values_next_state_target_model_tf = self.target_model(
            next_states_tf, training=False)

        # CAMBIO: Cálculo de Q-targets vectorizado usando TensorFlow y NumPy
        # Convertimos a NumPy para facilitar algunas indexaciones y np.amax,
        # pero idealmente, si todo el cálculo de targets se hace con tf ops,
        # se podría mantener todo en tensores para potencialmente más optimización.
        # Por ahora, esta mezcla es una buena mejora.

        # Empezar con los Q actuales (copia)
        q_targets_batch_np = q_values_current_state_tf.numpy()

        # Máximos Q-values para los siguientes estados (del target_model)
        q_values_next_max_np = np.amax(
            q_values_next_state_target_model_tf.numpy(), axis=1)

        # Q(s,a) = r  (si done)
        # Q(s,a) = r + gamma * max_a'(Q_target(s',a')) (si not done)
        updated_q_values_for_actions_taken = rewards_np + \
            self.gamma * q_values_next_max_np * (~dones_np)

        # Para los estados que son 'done', el target es solo la recompensa inmediata
        updated_q_values_for_actions_taken[dones_np] = rewards_np[dones_np]

        # Asignar estos targets calculados solo a las acciones que se tomaron
        batch_indices = np.arange(self.batch_size)
        q_targets_batch_np[batch_indices,
                           actions_np] = updated_q_values_for_actions_taken

        # Convertir los targets finales a Tensor para model.fit()
        q_targets_batch_tf = tf.convert_to_tensor(
            q_targets_batch_np, dtype=tf.float32)

        # Entrenar el modelo principal
        # Aquí states_tf y q_targets_batch_tf ya son tensores,
        # por lo que la conversión interna de Keras debería ser mínima.
        self.model.fit(states_tf, q_targets_batch_tf, epochs=10, verbose=0,
                       batch_size=self.batch_size)  # Especificar batch_size aquí también

    def load_model(self, filepath):
        try:
            # Para .keras o formato SavedModel, se usa load_model
            # Para .weights.h5, se usa load_weights
            # Asumiendo que el modelo completo se guarda
            if filepath.endswith(".keras"):
                self.model = tf.keras.models.load_model(
                    filepath)  # type: ignore
                print(
                    f"Modelo completo cargado desde {filepath} (incluye optimizador)")
            elif filepath.endswith(".weights.h5"):
                self.model.load_weights(filepath)  # Solo carga pesos
                print(f"Pesos del modelo cargados desde {filepath}")
            else:  # Intenta como si fuera un SavedModel (directorio)
                self.model = tf.keras.models.load_model(
                    filepath)  # type: ignore
                print(f"Modelo completo (SavedModel) cargado desde {filepath}")

            self.update_target_model()  # Sincronizar target model después de cargar

            # Considerar cargar/guardar el estado de epsilon si es importante
        except Exception as e:
            print(
                f"Error al cargar el modelo desde '{filepath}': {e}. Usando un modelo nuevo.")

    def save_model(self, filepath):
        try:
            # Para guardar solo pesos:
            self.model.save_weights(filepath)
            print(f"Pesos del modelo guardados en {filepath}")
            # Para guardar el modelo completo (arquitectura, pesos, estado del optimizador):
            # Asegúrate de que filepath termine en .keras o sea un directorio para SavedModel
            # if filepath.endswith(".keras"):
            #     self.model.save(filepath)
            #     print(f"Modelo completo guardado en {filepath}")
            # else: # Guardar como SavedModel (directorio)
            #     self.model.save(filepath) # filepath sería un nombre de directorio
            #     print(f"Modelo completo (SavedModel) guardado en {filepath}")
        except Exception as e:
            print(f"Error al guardar el modelo en '{filepath}': {e}")

    def reset_epsilon(self, epsilon_start=None):
        # CAMBIO: Si epsilon_start es None, usa el epsilon con el que se inicializó el agente
        # o un valor por defecto si ese también fuera None (aunque no debería serlo aquí).
        # Para ser más robusto, podríamos tener un self.epsilon_initial guardado en __init__.
        if epsilon_start is not None:
            self.epsilon = epsilon_start
        else:
            # Si no se provee epsilon_start, podrías querer resetearlo a un valor inicial guardado
            # o simplemente no hacer nada si la intención es continuar con el epsilon actual.
            # Para este ejemplo, lo resetearé al epsilon_start original si no se pasa nada.
            # Necesitarías guardar EPSILON_START_DQN en el __init__ del agente para esto
            # self.epsilon = self.initial_epsilon # Suponiendo que guardaste el epsilon inicial
            pass  # O simplemente no cambiarlo si epsilon_start es None


if __name__ == '__main__':
    # Ejemplo simple de uso
    STATE_SIZE_EXAMPLE = 9
    ACTION_SIZE_EXAMPLE = 4

    # tf.debugging.set_log_device_placement(True) # Para depurar ubicación de ops

    # Batch más pequeño para prueba rápida
    agent = DQNAgent(STATE_SIZE_EXAMPLE, ACTION_SIZE_EXAMPLE, batch_size=32)

    # Simular una transición
    # Asegurar que el estado es una tupla/lista de números, no un array de numpy ya aplanado aquí
    example_state_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
    # get_action espera el estado "crudo"
    example_action = agent.get_action(example_state_tuple)
    example_reward = 10
    example_next_state_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
    example_done = False

    agent.remember(example_state_tuple, example_action, example_reward,
                   example_next_state_tuple, example_done)
    print(f"Memoria después de una transición: {len(agent.memory)}")

    # Simular varias transiciones para tener suficiente para un lote
    for _ in range(agent.batch_size * 2):  # Usar agent.batch_size para consistencia
        s_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
        a = agent.get_action(s_tuple)
        # Recompensa como float32
        r = np.random.randint(-1, 2).astype(np.float32)
        s_next_tuple = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
        d = random.choice([True, False])
        agent.remember(s_tuple, a, r, s_next_tuple, d)

    print(f"Memoria antes de replay: {len(agent.memory)}")
    if len(agent.memory) >= agent.batch_size:
        agent.replay()
        print("Replay ejecutado.")
    else:
        print("No hay suficientes muestras en memoria para replay.")

    agent.update_target_model()
    print("Modelo objetivo actualizado.")

    # Guardar y cargar
    # O test_dqn_model.keras si guardas el modelo completo
    test_model_path = "test_dqn_model.weights.h5"
    agent.save_model(test_model_path)
    agent.load_model(test_model_path)
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
