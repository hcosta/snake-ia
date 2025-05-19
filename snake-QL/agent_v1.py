# agent_v1.py (clase Agent)

import random
import numpy as np
import pickle  # Importamos el módulo pickle
import os  # Para comprobar si el fichero existe


class Agent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05, q_table_filepath=None):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon_init = epsilon  # Guardamos el epsilon inicial para reinicios
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table_filepath = q_table_filepath
        self.q_table = {}

        if self.q_table_filepath:
            # Intentar cargar al iniciar
            self.load_q_table(self.q_table_filepath)

        # print("Agente Q-Learning inicializado...")

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def get_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self._get_q_values(state)
            action = self.actions[np.argmax(q_values)]
        return action

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def train(self, state, action, reward, next_state, done):
        q_values_state = self._get_q_values(state)
        q_values_next_state = self._get_q_values(next_state)
        try:
            action_index = self.actions.index(action)
        except ValueError:
            # print(f"Error: La acción {action} no está en la lista de acciones {self.actions}")
            return

        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(q_values_next_state)

        current_q_value = q_values_state[action_index]
        new_q_value = current_q_value + self.lr * \
            (reward + self.gamma * max_future_q - current_q_value)
        self.q_table[state][action_index] = new_q_value

    def save_q_table(self, filepath):
        """ Guarda la Tabla Q y el epsilon actual en un fichero usando pickle. """
        try:
            data_to_save = {
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }
            with open(filepath, 'wb') as f:  # 'wb' es para escribir en modo binario
                pickle.dump(data_to_save, f)
            print(f"Tabla Q y epsilon guardados en {filepath}")
        except Exception as e:
            print(f"Error al guardar la Tabla Q: {e}")

    def load_q_table(self, filepath):
        """ Carga la Tabla Q y el epsilon desde un fichero usando pickle. """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:  # 'rb' es para leer en modo binario
                    loaded_data = pickle.load(f)
                    # Usar {} si no se encuentra
                    self.q_table = loaded_data.get('q_table', {})
                    # Usar epsilon inicial si no se encuentra
                    self.epsilon = loaded_data.get(
                        'epsilon', self.epsilon_init)
                print(f"Tabla Q y epsilon cargados desde {filepath}")
                print(f"  Estados cargados: {len(self.q_table)}")
                print(f"  Epsilon cargado: {self.epsilon}")
            else:
                print(
                    f"Fichero de Tabla Q no encontrado en {filepath}. Se usará una tabla vacía y epsilon inicial.")
                self.q_table = {}  # Asegurar que la tabla está vacía si el fichero no existe
                self.epsilon = self.epsilon_init  # Usar epsilon inicial
        except Exception as e:
            print(
                f"Error al cargar la Tabla Q: {e}. Se usará una tabla vacía y epsilon inicial.")
            self.q_table = {}
            self.epsilon = self.epsilon_init

    def reset_epsilon(self):
        """ Reinicia epsilon a su valor inicial (útil para empezar un nuevo entrenamiento). """
        self.epsilon = self.epsilon_init


# Ejemplo de cómo podríamos usarlo:
if __name__ == '__main__':
    acciones = [0, 1, 2, 3]
    # .pkl es una extensión común para ficheros pickle
    ruta_fichero_q_table = "trained_data/snake_q_table.pkl"

    # Crear un agente
    # Intentará cargar si existe
    agente_1 = Agent(actions=acciones, q_table_filepath=ruta_fichero_q_table)

    # Simular algo de aprendizaje (llenar un poco la tabla)
    # Si no se cargó nada (fichero no existía o estaba vacío)
    if not agente_1.q_table:
        print(
            "\nSimulando algo de aprendizaje para agente_1 ya que la tabla estaba vacía...")
        agente_1.train(("estado1",), 0, 10, ("estado2",), False)
        agente_1.train(("estado2",), 1, -5, ("estado1",), False)
        agente_1.epsilon = 0.5  # Simular que epsilon ha decaído
        print(
            f"Tabla Q de agente_1 después de simular aprendizaje: {agente_1.q_table}")
        print(f"Epsilon de agente_1: {agente_1.epsilon}")
        # Guardar la tabla
        agente_1.save_q_table(ruta_fichero_q_table)
    else:
        print("\nAgente_1 cargó una tabla Q existente.")
        print(f"Tabla Q de agente_1: {agente_1.q_table}")
        print(f"Epsilon de agente_1: {agente_1.epsilon}")

    print("\n--- Creando un nuevo agente (agente_2) e intentando cargar la misma tabla ---")
    # Crear otro agente e intentar cargar la tabla guardada
    agente_2 = Agent(actions=acciones, q_table_filepath=ruta_fichero_q_table)

    if agente_2.q_table:
        print("¡Agente_2 cargó la Tabla Q guardada por agente_1 con éxito!")
        print(f"Tabla Q de agente_2: {agente_2.q_table}")
        print(f"Epsilon de agente_2: {agente_2.epsilon}")
    else:
        print("Agente_2 no pudo cargar la tabla (o estaba vacía).")

    # Opcional: Limpiar el fichero creado para la próxima ejecución del ejemplo
    import os
    if os.path.exists(ruta_fichero_q_table):
        os.remove(ruta_fichero_q_table)
        print(f"\nFichero {ruta_fichero_q_table} eliminado para limpieza.")
