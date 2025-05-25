# snake_logic_v14.py

"""
DDQN
"""

import random
import numpy as np
# Numba ya no es necesario si se elimina el BFS para tail_accessibility
from numba import jit, int32
# from numba.typed import List as NumbaList

# --- Constantes del Juego ---
SCREEN_WIDTH_LOGIC = 401  # Ancho lógico del tablero
SCREEN_HEIGHT_LOGIC = 401  # Alto lógico del tablero
STEP_SIZE_LOGIC = 20      # Tamaño de cada celda y paso de la serpiente

# CAMBIO: Acciones relativas según el informe.
# 0: Seguir recto
# 1: Girar a la izquierda (relativo a la dirección actual)
# 2: Girar a la derecha (relativo a la dirección actual)
# Ya no se usa ACTION_MAP_LOGIC directamente para el agente,
# la lógica de 'step' interpretará estos índices.

# --- Recompensas Alineadas con el Informe ---
REWARD_FOOD_LOGIC = 20
REWARD_WALL_COLLISION_LOGIC = -5
REWARD_SELF_COLLISION_LOGIC = -5
REWARD_MOVE_LOGIC = -0.05  # premiar cuanto más tiempo viva la serpiente
REWARD_STAGNATION_LOOP_LOGIC = -5  # Penalización si no come en muchos pasos.
REWARD_DANGER_MOVE = -1

# --- Funciones Auxiliares (Numba opcional si se mantiene para colisiones) ---


@jit(nopython=True, cache=True)  # Descomentar si se usa Numba
def _is_wall_collision_static(x: int, y: int, width: int, height: int, step_size: int) -> bool:
    """Comprueba si las coordenadas (x, y) colisionan con los límites del tablero."""
    half_step = step_size // 2
    # Comprueba si el centro de la cabeza está fuera de los límites permitidos.
    # Se ajusta por half_step para considerar el centro de la celda.
    # El `+ step_size % 2` maneja el caso de step_size impar, aunque usualmente es par.
    if not (half_step <= x < width - half_step + step_size % 2 and
            half_step <= y < height - half_step + step_size % 2):
        return True
    return False


@jit(nopython=True, cache=True)  # Descomentar si se usa Numba
def _is_body_collision_static(x: int, y: int, snake_body_np: np.ndarray) -> bool:
    """Comprueba si las coordenadas (x, y) colisionan con el cuerpo de la serpiente."""
    for i in range(snake_body_np.shape[0]):
        if x == snake_body_np[i, 0] and y == snake_body_np[i, 1]:
            return True
    return False


class SnakeLogic:
    def __init__(self, width, height, step_size):
        self.width = width
        self.height = height
        self.step_size = step_size

        self.snake_body = []  # Lista de diccionarios {'x': val, 'y': val}
        self.current_dx = 0   # Dirección actual en X
        self.current_dy = 0   # Dirección actual en Y

        self.food_x = 0
        self.food_y = 0
        self.score = 0
        self.game_over = False

        # CAMBIO: Contador de pasos sin comer para la penalización por estancamiento/bucle.
        self.steps_since_last_food = 0
        self.max_steps_without_food = (self.width // self.step_size) * \
                                      (self.height // self.step_size) * \
            1.5  # Umbral heurístico

        self.setup()

    def setup(self):
        """Configura o resetea el estado inicial del juego."""
        self.game_over = False
        self.score = 0
        self.steps_since_last_food = 0  # CAMBIO: Resetear contador

        # Calcula la posición inicial de la cabeza arriba a la izquierda (celda 1,1).
        start_cell_col = 1
        start_cell_row = 1
        head_start_x = start_cell_col * self.step_size + (self.step_size // 2)
        head_start_y = start_cell_row * self.step_size + (self.step_size // 2)

        self.snake_body = [
            {'x': head_start_x, 'y': head_start_y},
            # Segmento detrás de la cabeza
            {'x': head_start_x - self.step_size, 'y': head_start_y},
            {'x': head_start_x - 2 * self.step_size,
                'y': head_start_y}  # Tercer segmento
        ]
        # CAMBIO: Dirección inicial por defecto (ej: moviéndose a la derecha).
        # El informe no especifica, pero es necesario para las acciones relativas.
        self.current_dx = self.step_size
        self.current_dy = 0

        self._place_food()
        return self.get_state()

    def reset(self):
        """Resetea el juego a su estado inicial."""
        return self.setup()

    def _place_food(self):
        """Coloca la comida en una posición aleatoria que no esté ocupada por la serpiente."""
        placed_on_snake = True
        while placed_on_snake:
            # Genera coordenadas para la comida (centro de una celda).
            self.food_x = random.randrange(
                0, self.width // self.step_size) * self.step_size + (self.step_size // 2)
            self.food_y = random.randrange(
                0, self.height // self.step_size) * self.step_size + (self.step_size // 2)

            placed_on_snake = False
            for segment in self.snake_body:
                if segment['x'] == self.food_x and segment['y'] == self.food_y:
                    placed_on_snake = True
                    break
        self.steps_since_last_food = 0  # CAMBIO: Resetear contador al colocar nueva comida

    def _is_wall_collision(self, x, y):
        """Comprueba colisión con la pared usando la función estática."""
        return _is_wall_collision_static(x, y, self.width, self.height, self.step_size)

    def _is_body_collision(self, x, y, check_body_from_index=0):
        """Comprueba colisión con el cuerpo de la serpiente."""
        if not self.snake_body or check_body_from_index >= len(self.snake_body):
            return False
        # Considera solo los segmentos relevantes del cuerpo para la colisión.
        body_to_check_list = self.snake_body[check_body_from_index:]
        if not body_to_check_list:
            return False
        # Convierte a NumPy array para la función estática si se usa Numba.
        # Si no se usa Numba, se puede iterar directamente sobre body_to_check_list.
        snake_body_np = np.array([[seg['x'], seg['y']]
                                 for seg in body_to_check_list], dtype=np.int32)
        return _is_body_collision_static(x, y, snake_body_np)

    def _is_general_collision(self, x, y, check_body_from_index=0):
        """Comprueba colisión general (pared o cuerpo)."""
        if self._is_wall_collision(x, y):
            return True
        # Para peligro inmediato, se suele comprobar colisión con el cuerpo a partir del segmento 1 (evitando la cabeza actual).
        if self._is_body_collision(x, y, check_body_from_index):
            return True
        return False

    # CAMBIO: Función get_state completamente redefinida para 11 parámetros booleanos.
    def get_state(self):
        """
        Calcula el estado actual del juego como una tupla de 11 booleanos (0 o 1).
        Los parámetros están alineados con el informe de referencia.
        """
        if not self.snake_body:  # Caso improbable, pero por seguridad
            return (0,) * 11

        head = self.snake_body[0]

        # Determinar la dirección efectiva actual (dx_eff, dy_eff)
        # Esto es importante porque current_dx y current_dy reflejan la *última* acción tomada.
        dx_eff, dy_eff = self.current_dx, self.current_dy
        # Si la serpiente aún no se ha movido (al inicio del juego, current_dx/dy podrían ser 0,0)
        # o si por alguna razón se detuvo, se infiere la dirección del segmento anterior.
        if dx_eff == 0 and dy_eff == 0:
            if len(self.snake_body) > 1:
                prev_segment = self.snake_body[1]
                # Diferencia de coordenadas para inferir dirección
                dx_eff = np.sign(
                    head['x'] - prev_segment['x']) * self.step_size
                dy_eff = np.sign(
                    head['y'] - prev_segment['y']) * self.step_size
                # Si cabeza y segmento anterior coinciden (improbable)
                if dx_eff == 0 and dy_eff == 0:
                    dx_eff = self.step_size  # Por defecto, moverse a la derecha
            else:  # Serpiente de un solo segmento
                dx_eff = self.step_size  # Por defecto, moverse a la derecha

        # 1. Peligro Recto (en la dirección actual dx_eff, dy_eff)
        #    Coordenadas del punto directamente en frente de la cabeza.
        point_straight_x = head['x'] + dx_eff
        point_straight_y = head['y'] + dy_eff
        danger_straight = 1 if self._is_general_collision(
            point_straight_x, point_straight_y, 1) else 0

        # 2. Peligro a la Izquierda Relativa
        #    Dirección a la izquierda relativa: dx_left_rel = -dy_eff, dy_left_rel = dx_eff
        dx_left_rel = -dy_eff
        dy_left_rel = dx_eff
        point_left_x = head['x'] + dx_left_rel
        point_left_y = head['y'] + dy_left_rel
        danger_left_relative = 1 if self._is_general_collision(
            point_left_x, point_left_y, 1) else 0

        # 3. Peligro a la Derecha Relativa
        #    Dirección a la derecha relativa: dx_right_rel = dy_eff, dy_right_rel = -dx_eff
        dx_right_rel = dy_eff
        dy_right_rel = -dx_eff
        point_right_x = head['x'] + dx_right_rel
        point_right_y = head['y'] + dy_right_rel
        danger_right_relative = 1 if self._is_general_collision(
            point_right_x, point_right_y, 1) else 0

        # 4-7. Dirección actual de movimiento (basada en dx_eff, dy_eff)
        # Asumiendo: +Y es ARRIBA, -Y es ABAJO, -X es IZQUIERDA, +X es DERECHA
        # Ajustar si el sistema de coordenadas es diferente (PyGame suele tener +Y hacia abajo)
        # En este código, ACTION_MAP_LOGIC (si se usara) tenía +Y para UP.
        # Si current_dy es positivo, se mueve hacia "arriba" en la lógica.
        moving_up = 1 if dy_eff > 0 else 0  # Y aumenta
        moving_down = 1 if dy_eff < 0 else 0  # Y disminuye
        moving_left = 1 if dx_eff < 0 else 0  # X disminuye
        moving_right = 1 if dx_eff > 0 else 0  # X aumenta

        # 8-11. Posición de la comida relativa a la cabeza de la serpiente
        food_left_of_snake = 1 if self.food_x < head['x'] else 0
        food_right_of_snake = 1 if self.food_x > head['x'] else 0
        # Asumiendo +Y es arriba
        food_above_snake = 1 if self.food_y > head['y'] else 0
        # Asumiendo +Y es arriba
        food_below_snake = 1 if self.food_y < head['y'] else 0

        state = (
            danger_straight,
            danger_left_relative,
            danger_right_relative,
            moving_up,
            moving_down,
            moving_left,
            moving_right,
            food_left_of_snake,
            food_right_of_snake,
            food_above_snake,
            food_below_snake
        )
        return state

    # CAMBIO: Lógica de 'step' para manejar acciones relativas y penalización por bucle.
    def step(self, relative_action_index):
        """
        Ejecuta un paso en el juego basado en una acción relativa.
        relative_action_index: 0 (recto), 1 (izquierda rel.), 2 (derecha rel.)
        """
        reward = REWARD_MOVE_LOGIC  # Recompensa base por moverse
        self.game_over = False
        info = {'collision_type': None, 'ate_food': False, 'stagnated': False}

        # Obtener estado actual y peligros
        current_state = self.get_state()
        danger_straight, danger_left, danger_right = current_state[
            0], current_state[1], current_state[2]

        # Determinar la dirección efectiva actual (dx_eff, dy_eff)
        # Similar a get_state, para asegurar que hay una dirección base para las acciones relativas.
        dx_eff, dy_eff = self.current_dx, self.current_dy
        if dx_eff == 0 and dy_eff == 0:  # Si no hay movimiento previo (inicio)
            if len(self.snake_body) > 1:
                prev_segment = self.snake_body[1]
                dx_eff = np.sign(
                    self.snake_body[0]['x'] - prev_segment['x']) * self.step_size
                dy_eff = np.sign(
                    self.snake_body[0]['y'] - prev_segment['y']) * self.step_size
                if dx_eff == 0 and dy_eff == 0:
                    dx_eff = self.step_size  # Default
            else:
                dx_eff = self.step_size  # Default

        # Calcular la nueva dirección (requested_dx, requested_dy) basada en la acción relativa
        requested_dx, requested_dy = dx_eff, dy_eff  # Por defecto, seguir recto

        if relative_action_index == 1:  # Girar a la izquierda
            requested_dx = -dy_eff  # Nueva dx es -vieja dy
            requested_dy = dx_eff   # Nueva dy es vieja dx
        elif relative_action_index == 2:  # Girar a la derecha
            requested_dx = dy_eff   # Nueva dx es vieja dy
            requested_dy = -dx_eff  # Nueva dy es -vieja dx
        # Si relative_action_index == 0 (recto), requested_dx/dy ya son dx_eff/dy_eff

        # Penalizar acción si se mueve hacia un peligro
        if relative_action_index == 0 and danger_straight:
            reward += REWARD_DANGER_MOVE
        elif relative_action_index == 1 and danger_left:
            reward += REWARD_DANGER_MOVE
        elif relative_action_index == 2 and danger_right:
            reward += REWARD_DANGER_MOVE

        # Actualizar la dirección actual de la serpiente para el próximo estado y get_state
        self.current_dx = requested_dx
        self.current_dy = requested_dy

        # Mover la cabeza de la serpiente
        head = self.snake_body[0]
        new_head_x = head['x'] + self.current_dx
        new_head_y = head['y'] + self.current_dy

        # Comprobar colisiones
        if self._is_wall_collision(new_head_x, new_head_y):
            self.game_over = True
            reward += REWARD_WALL_COLLISION_LOGIC
            info['collision_type'] = "wall"
        # Comprobar con todo el cuerpo (índice 0)
        elif self._is_body_collision(new_head_x, new_head_y, 0):
            self.game_over = True
            reward += REWARD_SELF_COLLISION_LOGIC
            info['collision_type'] = "self"

        if self.game_over:
            return self.get_state(), reward, self.game_over, info

        # Si no hay colisión, actualizar el cuerpo de la serpiente
        new_head_segment = {'x': new_head_x, 'y': new_head_y}
        ate_food_this_step = False

        if new_head_x == self.food_x and new_head_y == self.food_y:
            ate_food_this_step = True
            self.score += 1
            reward += REWARD_FOOD_LOGIC
            info['ate_food'] = True
            self.steps_since_last_food = 0  # Resetear contador
        else:
            self.steps_since_last_food += 1  # Incrementar contador

        self.snake_body.insert(0, new_head_segment)  # Añadir nueva cabeza

        if not ate_food_this_step:
            self.snake_body.pop()  # Quitar cola si no comió
        else:
            self._place_food()  # Colocar nueva comida si comió (esto ya resetea steps_since_last_food)

        # CAMBIO: Comprobar estancamiento/bucle
        if self.steps_since_last_food > self.max_steps_without_food:
            self.game_over = True  # Terminar el juego si se estanca
            reward += REWARD_STAGNATION_LOOP_LOGIC
            info['stagnated'] = True
            info['collision_type'] = "stagnation"  # Nuevo tipo de finalización

        return self.get_state(), reward, self.game_over, info
