# agent_v2.py (Agente DQN)
import random
import numpy as np
from collections import deque  # Para la memoria de repetición
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 replay_memory_size=10000, batch_size=64, model_filepath='dqn_snake_model.keras'):  # Nota la extensión .keras

        # Longitud del vector de estado (ej: 9 para tu estado)
        self.state_size = state_size
        self.action_size = action_size  # Número de acciones posibles (ej: 4)

        # Hiperparámetros del agente DQN
        self.gamma = discount_factor         # Factor de descuento
        self.epsilon = epsilon               # Probabilidad de exploración inicial
        self.epsilon_decay = epsilon_decay   # Tasa de decaimiento de epsilon
        self.epsilon_min = epsilon_min       # Epsilon mínimo
        self.learning_rate = learning_rate   # Tasa de aprendizaje para la red neuronal
        # Tamaño del lote para el entrenamiento de la red
        self.batch_size = batch_size

        self.model_filepath = model_filepath

        # Memoria de repetición (experience replay)
        # deque es una lista doblemente enlazada, eficiente para añadir y quitar elementos
        self.memory = deque(maxlen=replay_memory_size)

        # Modelo principal (el que se entrena y toma decisiones)
        self.model = self._build_model()
        # Modelo objetivo (target network, para estabilizar el aprendizaje)
        self.target_model = self._build_model()
        self.update_target_model()  # Sincronizar pesos al inicio

        if os.path.exists(self.model_filepath):
            self.load_model(self.model_filepath)
            print(f"Modelo cargado desde {self.model_filepath}")
        else:
            print(
                f"No se encontró un modelo guardado en {self.model_filepath}. Iniciando con un nuevo modelo.")

    def _build_model(self):
        """Construye la red neuronal para aproximar Q(s,a)."""
        model = Sequential()
        # La capa de entrada debe coincidir con state_size
        # Estas son capas densas (fully connected). Podrías experimentar con más capas o neuronas.
        # Primera capa oculta
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))  # Segunda capa oculta
        # La capa de salida tiene 'action_size' neuronas, una para cada acción posible.
        # Usa activación lineal porque queremos predecir los valores Q directamente.
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',  # Mean Squared Error es común para regresión (predecir Q-values)
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copia los pesos del modelo principal al modelo objetivo."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Almacena una transición (experiencia) en la memoria de repetición."""
        # El estado y next_state deben ser arrays de numpy para la red
        state = np.array(state)
        next_state = np.array(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """Elige una acción usando la política epsilon-greedy."""
        if random.random() <= self.epsilon:
            # Acción aleatoria (índice de la acción)
            return random.randrange(self.action_size)
        else:
            # El estado debe ser un array de numpy y tener la forma (1, state_size) para la predicción
            state_tensor = np.reshape(state, [1, self.state_size])
            # verbose=0 para no imprimir salida
            act_values = self.model.predict(state_tensor, verbose=0)
            # Elige la acción con el mayor Q-value (índice)
            return np.argmax(act_values[0])

    def replay(self):
        """Entrena la red neuronal usando un lote de experiencias de la memoria."""
        if len(self.memory) < self.batch_size:
            return  # No entrenar si no hay suficientes experiencias

        # Muestrear un lote aleatorio de la memoria
        minibatch = random.sample(self.memory, self.batch_size)

        # Preparamos los datos para el entrenamiento por lotes para eficiencia
        states = np.array([experience[0].flatten()
                          for experience in minibatch])  # (batch_size, state_size)
        # (batch_size, state_size)
        next_states = np.array([experience[3].flatten()
                               for experience in minibatch])

        # Predecir Q-values para los estados actuales y siguientes usando los modelos
        # Usamos model.predict para los Q-values de las acciones tomadas
        # Usamos target_model.predict para los Q-values de las mejores acciones futuras (estabilidad)
        q_values_current_state = self.model.predict(states, verbose=0)
        q_values_next_state_target_model = self.target_model.predict(
            next_states, verbose=0)

        # Actualizar los Q-values para el lote
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                # Ecuación de Bellman: Q(s,a) = r + gamma * max_a'(Q_target(s',a'))
                target = reward + self.gamma * \
                    np.amax(q_values_next_state_target_model[i])

            # El target para la acción tomada es el nuevo Q-value calculado
            # Los targets para las otras acciones se mantienen como los predichos por self.model
            # para no afectar sus gradientes innecesariamente.
            q_values_current_state[i][action] = target

        # Entrenar el modelo principal con los estados y los targets Q actualizados
        self.model.fit(states, q_values_current_state,
                       epochs=1, verbose=0)  # epochs=1 por lote

        # Actualizar epsilon (reducir exploración)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, filepath):
        try:
            self.model.load_weights(filepath)
            # Asegurar que target_model también se carga
            self.target_model.load_weights(filepath)
            # Cargar epsilon podría guardarse en un archivo separado o no guardarse y empezar de nuevo la exploración
            print(f"Pesos del modelo cargados desde {filepath}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}. Usando un modelo nuevo.")

    def save_model(self, filepath):
        try:
            self.model.save_weights(filepath)
            print(f"Pesos del modelo guardados en {filepath}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")

    def reset_epsilon(self, epsilon_start=None):
        """ Reinicia epsilon a su valor inicial (útil para empezar un nuevo entrenamiento). """
        self.epsilon = epsilon_start if epsilon_start is not None else self.epsilon  # O un valor por defecto


if __name__ == '__main__':
    # Ejemplo simple de uso
    STATE_SIZE_EXAMPLE = 9  # Tu estado tiene 9 características
    ACTION_SIZE_EXAMPLE = 4  # Arriba, Abajo, Izquierda, Derecha

    agent = DQNAgent(STATE_SIZE_EXAMPLE, ACTION_SIZE_EXAMPLE)

    # Simular una transición
    example_state = tuple(np.random.rand(
        STATE_SIZE_EXAMPLE))  # Estado aleatorio
    example_action = agent.get_action(example_state)
    example_reward = 10
    example_next_state = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
    example_done = False

    agent.remember(example_state, example_action, example_reward,
                   example_next_state, example_done)
    print(f"Memoria después de una transición: {len(agent.memory)}")

    # Simular varias transiciones para tener suficiente para un lote
    for _ in range(agent.batch_size * 2):
        s = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
        a = agent.get_action(s)
        r = np.random.randint(-1, 2)
        s_next = tuple(np.random.rand(STATE_SIZE_EXAMPLE))
        d = random.choice([True, False])
        agent.remember(s, a, r, s_next, d)

    print(f"Memoria antes de replay: {len(agent.memory)}")
    agent.replay()  # Entrenar la red
    print("Replay ejecutado.")
    agent.update_target_model()  # Actualizar el modelo objetivo
    print("Modelo objetivo actualizado.")

    # Guardar y cargar (opcional, solo para probar)
    # agent.save_model("test_dqn_model.weights.h5") # Keras >2.6 prefiere .weights.h5 o .keras
    # agent.load_model("test_dqn_model.weights.h5")
    # if os.path.exists("test_dqn_model.weights.h5"):
    #     os.remove("test_dqn_model.weights.h5")
