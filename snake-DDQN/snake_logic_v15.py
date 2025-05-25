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
STATE_SIZE_EXPECTED = 15

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

    # Dentro de la clase SnakeLogic en snake_logic_v15.py
    def _is_immediate_trap(self, hypothetical_head_x, hypothetical_head_y,
                           dx_at_hypothetical_pos, dy_at_hypothetical_pos):
        """
        Evalúa si la posición (hypothetical_head_x, hypothetical_head_y)
        con la dirección (dx_at_hypothetical_pos, dy_at_hypothetical_pos)
        es una trampa donde todos los movimientos relativos siguientes son colisiones.

        Para esta comprobación, el "cuerpo" relevante es el cuerpo de la serpiente
        COMO SI YA SE HUBIERA MOVIDO a (hypothetical_head_x, hypothetical_head_y).
        """
        if not self.snake_body:  # No debería ocurrir si se llama desde get_state
            return 0

        # 1. Construir el cuerpo hipotético de la serpiente
        #    La nueva cabeza está en (hypothetical_head_x, hypothetical_head_y)
        #    El resto del cuerpo son los segmentos anteriores de la cabeza actual,
        #    asumiendo que la serpiente se movió un paso sin comer.
        hypothetical_body_list = [
            {'x': hypothetical_head_x, 'y': hypothetical_head_y}]
        # Tomamos los primeros N-1 segmentos del cuerpo actual para formar la cola del cuerpo hipotético
        # donde N es la longitud actual de la serpiente.
        # Esto simula que la cabeza se movió y el resto del cuerpo la siguió.
        for i in range(len(self.snake_body) - 1):
            hypothetical_body_list.append(self.snake_body[i])

        # Si la serpiente original solo tenía 1 segmento (la cabeza),
        # el cuerpo hipotético solo tendrá la hypothetical_head.
        # Esto está bien, ya que no puede chocar consigo misma si solo mide 1.

        hypothetical_snake_body_np = np.array(
            [[seg['x'], seg['y']] for seg in hypothetical_body_list], dtype=np.int32
        )

        # 2. Evaluar las 3 acciones relativas DESDE la posición hipotética

        #   a. Moverse Recto desde la posición hipotética
        next_potential_x_s = hypothetical_head_x + dx_at_hypothetical_pos
        next_potential_y_s = hypothetical_head_y + dy_at_hypothetical_pos

        is_wall_s = self._is_wall_collision(
            next_potential_x_s, next_potential_y_s)
        # Para la colisión con el cuerpo, usamos el cuerpo hipotético.
        # (s2_x, s2_y) no puede ser igual a (hypothetical_head_x, hypothetical_head_y) porque es un paso adelante.
        # Pero sí puede chocar con el resto del cuerpo hipotético.
        is_body_s = False
        if not is_wall_s:  # Solo chequear cuerpo si no hay pared
            is_body_s = _is_body_collision_static(
                next_potential_x_s, next_potential_y_s, hypothetical_snake_body_np)
        danger_s = is_wall_s or is_body_s

        #   b. Girar a la Izquierda desde la posición hipotética
        dx_l = -dy_at_hypothetical_pos
        dy_l = dx_at_hypothetical_pos
        next_potential_x_l = hypothetical_head_x + dx_l
        next_potential_y_l = hypothetical_head_y + dy_l

        is_wall_l = self._is_wall_collision(
            next_potential_x_l, next_potential_y_l)
        is_body_l = False
        if not is_wall_l:
            is_body_l = _is_body_collision_static(
                next_potential_x_l, next_potential_y_l, hypothetical_snake_body_np)
        danger_l = is_wall_l or is_body_l

        #   c. Girar a la Derecha desde la posición hipotética
        dx_r = dy_at_hypothetical_pos
        dy_r = -dx_at_hypothetical_pos
        next_potential_x_r = hypothetical_head_x + dx_r
        next_potential_y_r = hypothetical_head_y + dy_r

        is_wall_r = self._is_wall_collision(
            next_potential_x_r, next_potential_y_r)
        is_body_r = False
        if not is_wall_r:
            is_body_r = _is_body_collision_static(
                next_potential_x_r, next_potential_y_r, hypothetical_snake_body_np)
        danger_r = is_wall_r or is_body_r

        if danger_s and danger_l and danger_r:
            return 1  # Es una trampa: todos los movimientos desde la posición hipotética son peligrosos
        return 0
    # CAMBIO: Función get_state completamente redefinida para 11 parámetros booleanos.

    def _is_head_critically_near_distant_tail(self, critical_manhattan_distance=2):
        """
        ¿Se Está Cerrando un Bucle Sobre Sí Misma?
        Concepto: Esta bandera se activaría (True/1) si la cabeza de la serpiente está muy cerca de un segmento "distante" de su propia cola, de una manera que sugiere que está a punto de encerrarse en un bucle o que el espacio se está volviendo críticamente restringido por su propia cola. No se trata de un choque inmediato con el cuello, sino de un peligro de auto-atrapamiento a corto-medio plazo.
        - Solo se considera si la serpiente tiene una longitud mínima (ej. > 4 o 5 segmentos) para que "cerrar un bucle" tenga sentido.
        - Se toma la posición de la cabeza.
        - Se itera sobre los segmentos de la cola, excluyendo los primeros segmentos cercanos a la cabeza (el "cuello", que ya están cubiertos por danger_X). Por ejemplo, empezar a comprobar desde el 4º o 5º segmento del cuerpo (índice 3 o 4).
        - Para cada uno de estos segmentos "distantes" de la cola, se calcula la distancia a la cabeza.
        - Si la cabeza está a una distancia críticamente pequeña (ej. 1 o 2 casillas de Manhattan, o un radio pequeño) de alguno de estos segmentos distantes de la cola, la bandera se activa.
        """
        if len(self.snake_body) < 5:  # Umbral de longitud para que tenga sentido
            return 0

        head_pos = self.snake_body[0]
        # Comprobar contra segmentos de la cola, saltándose el cuello (e.g., los 3 primeros después de la cabeza)
        for i in range(3, len(self.snake_body)):
            segment_pos = self.snake_body[i]
            dist_x = abs(head_pos['x'] - segment_pos['x'])
            dist_y = abs(head_pos['y'] - segment_pos['y'])

            # Distancia de Manhattan (en unidades de celda)
            manhattan_dist_cells = (dist_x + dist_y) / self.step_size

            if manhattan_dist_cells <= critical_manhattan_distance:
                return 1  # La cabeza está críticamente cerca de un segmento distante de la cola
        return 0

    def get_state(self):
        """
        Calcula el estado actual del juego como una tupla de 14 booleanos (0 o 1).
        Incluye peligros inmediatos y detección de trampas/callejones de 1 paso.
        """
        # El nuevo STATE_SIZE será 14
        if not self.snake_body:
            return (0,) * 14

        head = self.snake_body[0]
        dx_eff, dy_eff = self.current_dx, self.current_dy

        if dx_eff == 0 and dy_eff == 0:  # Determinar dirección efectiva si está parada
            if len(self.snake_body) > 1:
                prev_segment = self.snake_body[1]
                dx_eff = np.sign(
                    head['x'] - prev_segment['x']) * self.step_size
                dy_eff = np.sign(
                    head['y'] - prev_segment['y']) * self.step_size
                if dx_eff == 0 and dy_eff == 0:
                    dx_eff = self.step_size
            else:
                dx_eff = self.step_size

        # --- Peligros Inmediatos (1 paso adelante desde la posición ACTUAL) ---
        # Para estos, _is_general_collision usa check_body_from_index=1 para ignorar la cabeza actual.

        # Peligro Recto
        point_straight_x = head['x'] + dx_eff
        point_straight_y = head['y'] + dy_eff
        danger_straight = 1 if self._is_general_collision(
            point_straight_x, point_straight_y, 1) else 0

        # Peligro Izquierda Relativa
        # Dirección si gira a la izquierda DESDE la posición actual
        dx_left_rel_current = -dy_eff
        dy_left_rel_current = dx_eff
        point_left_x = head['x'] + dx_left_rel_current
        point_left_y = head['y'] + dy_left_rel_current
        danger_left_relative = 1 if self._is_general_collision(
            point_left_x, point_left_y, 1) else 0

        # Peligro Derecha Relativa
        # Dirección si gira a la derecha DESDE la posición actual
        dx_right_rel_current = dy_eff
        dy_right_rel_current = -dx_eff
        point_right_x = head['x'] + dx_right_rel_current
        point_right_y = head['y'] + dy_right_rel_current
        danger_right_relative = 1 if self._is_general_collision(
            point_right_x, point_right_y, 1) else 0

        # --- Detección de Callejones (evaluando 1 paso MÁS ALLÁ de un movimiento seguro) ---

        # Callejón si va Recto:
        # Se evalúa desde (point_straight_x, point_straight_y)
        # La dirección en esa casilla hipotética sería la misma que la actual (dx_eff, dy_eff)
        trap_straight = 0
        if not danger_straight:  # Solo chequear trampa si el primer movimiento es seguro
            trap_straight = self._is_immediate_trap(
                point_straight_x, point_straight_y, dx_eff, dy_eff)

        # Callejón si Gira a la Izquierda:
        # Se evalúa desde (point_left_x, point_left_y)
        # La dirección en esa casilla hipotética sería (dx_left_rel_current, dy_left_rel_current)
        trap_left = 0
        if not danger_left_relative:
            trap_left = self._is_immediate_trap(
                point_left_x, point_left_y, dx_left_rel_current, dy_left_rel_current)

        # Callejón si Gira a la Derecha:
        # Se evalúa desde (point_right_x, point_right_y)
        # La dirección en esa casilla hipotética sería (dx_right_rel_current, dy_right_rel_current)
        trap_right = 0
        if not danger_right_relative:
            trap_right = self._is_immediate_trap(
                point_right_x, point_right_y, dx_right_rel_current, dy_right_rel_current)

        # Está Cerrando un Bucle Sobre Sí Misma?
        head_near_distant_tail = self._is_head_critically_near_distant_tail()

        # --- Resto de las características del estado (sin cambios) ---
        moving_up = 1 if dy_eff > 0 else 0
        moving_down = 1 if dy_eff < 0 else 0
        moving_left = 1 if dx_eff < 0 else 0
        moving_right = 1 if dx_eff > 0 else 0

        food_left_of_snake = 1 if self.food_x < head['x'] else 0
        food_right_of_snake = 1 if self.food_x > head['x'] else 0
        # Asumiendo +Y es pantalla arriba
        food_above_snake = 1 if self.food_y > head['y'] else 0
        food_below_snake = 1 if self.food_y < head['y'] else 0

        assert len(
            state) == STATE_SIZE_EXPECTED, "La longitud del estado no coincide con la esperada (15)"

        state = (
            danger_straight,
            danger_left_relative,
            danger_right_relative,
            trap_straight,          # NUEVO
            trap_left,              # NUEVO
            trap_right,             # NUEVO
            head_near_distant_tail,  # NUEVO 2
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
