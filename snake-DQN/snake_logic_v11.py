# snake_logic_v11.py
import random
import numpy as np
from numba import jit  # Importar jit de Numba

# --- Constantes del Juego ---
SCREEN_WIDTH_LOGIC = 250
SCREEN_HEIGHT_LOGIC = 250
STEP_SIZE_LOGIC = 20

ACTION_MAP_LOGIC = {
    # UP (En lógica, Y aumenta hacia arriba) -> CUIDADO: En UI Y puede ser hacia abajo
    0: {'dx': 0, 'dy': STEP_SIZE_LOGIC},
    1: {'dx': 0, 'dy': -STEP_SIZE_LOGIC},   # DOWN
    2: {'dx': -STEP_SIZE_LOGIC, 'dy': 0},   # LEFT
    3: {'dx': STEP_SIZE_LOGIC, 'dy': 0}     # RIGHT
}
# Nota sobre coordenadas: Es crucial ser consistente. Si en la UI (Arcade) Y=0 es abajo,
# y en la lógica Y=0 es abajo, entonces UP debería ser dy = -STEP_SIZE y DOWN dy = STEP_SIZE.
# Voy a asumir que la lógica de ACTION_MAP_LOGIC es la canónica para el agente,
# y la UI se adapta si es necesario. Por los nombres de las teclas en la UI, parece que
# UP = Y aumenta, DOWN = Y disminuye, lo cual es común en coordenadas cartesianas.
# Sin embargo, el código de snake_ui_v10.py para arcade.key.UP usa dy = STEP_SIZE_LOGIC.
# Esto implica que en la lógica de Arcade (y por tanto en la visualización), Y positivo es hacia arriba.
# Mantendremos esta consistencia.

# --- Recompensas ---
REWARD_FOOD_LOGIC = 20
REWARD_WALL_COLLISION_LOGIC = -50
REWARD_SELF_COLLISION_LOGIC = -100
REWARD_MOVE_LOGIC = -1


# ---------------------------------------------------------------------------
# Numba helper functions (si es necesario para estructuras de datos complejas)
# ---------------------------------------------------------------------------
# Por ahora, intentaremos aplicar @jit directamente a los métodos.

class SnakeLogic:
    def __init__(self, width, height, step_size):
        self.width = width
        self.height = height
        self.step_size = step_size

        self.snake_body = []  # Lista de diccionarios {'x': x_val, 'y': y_val}
        # Para una optimización más profunda con Numba, self.snake_body
        # idealmente sería un array de NumPy, ej: np.zeros((max_len, 2), dtype=np.int32)
        # y se manejaría la longitud actual por separado.

        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self.food_x = 0
        self.food_y = 0
        self.score = 0
        self.game_over = False

        self.setup()

    def setup(self):
        self.game_over = False
        self.score = 0
        num_cols = self.width // self.step_size
        num_rows = self.height // self.step_size
        center_col = num_cols // 2
        center_row = num_rows // 2
        head_start_x = center_col * self.step_size + (self.step_size // 2)
        head_start_y = center_row * self.step_size + (self.step_size // 2)

        self.snake_body = [
            {'x': head_start_x, 'y': head_start_y},
            {'x': head_start_x - self.step_size, 'y': head_start_y},
            {'x': head_start_x - 2 * self.step_size, 'y': head_start_y}
        ]
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self._place_food()
        return self.get_state()

    def reset(self):
        return self.setup()

    def _place_food(self):
        # Esta función es candidata a Numba si el bucle 'while' o la comprobación
        # 'for segment in self.snake_body' se vuelven un cuello de botella.
        # Con self.snake_body como lista de dicts, Numba nopython=True fallaría aquí.
        # Si snake_body fuera un array NumPy, podríamos hacer una versión Numba.
        placed_on_snake = True
        while placed_on_snake:
            self.food_x = random.randrange(
                0, self.width // self.step_size) * self.step_size + (self.step_size // 2)
            self.food_y = random.randrange(
                0, self.height // self.step_size) * self.step_size + (self.step_size // 2)
            placed_on_snake = False
            for segment in self.snake_body:
                if segment['x'] == self.food_x and segment['y'] == self.food_y:
                    placed_on_snake = True
                    break

    # Esta función es un buen candidato para Numba ya que opera con tipos simples.
    @staticmethod
    @jit(nopython=True, cache=True)  # cache=True para guardar la compilación
    def _is_wall_collision_numba(x, y, width, height, step_size):
        half_step = step_size // 2
        # Ajuste para bordes: la coordenada (x,y) es el centro del segmento.
        # Un segmento está fuera si su centro está más allá de half_step del borde.
        if not (half_step <= x < width - half_step + step_size % 2 and
                half_step <= y < height - half_step + step_size % 2):
            return True
        return False

    def _is_wall_collision(self, x, y):
        return SnakeLogic._is_wall_collision_numba(x, y, self.width, self.height, self.step_size)

    # Para _is_body_collision, necesitamos pasar el cuerpo como un array NumPy a una función Numba.
    @staticmethod
    @jit(nopython=True, cache=True)
    def _is_body_collision_numba(x, y, snake_body_np, check_body_from_index):
        # snake_body_np es un array de Nx2 [[x1,y1], [x2,y2], ...]
        for i in range(check_body_from_index, snake_body_np.shape[0]):
            if x == snake_body_np[i, 0] and y == snake_body_np[i, 1]:
                return True
        return False

    def _is_body_collision(self, x, y, check_body_from_index=0):
        if not self.snake_body:  # Si el cuerpo está vacío
            return False
        # Convertir la parte relevante de snake_body a NumPy array para Numba
        # Esto tiene un coste, pero si el bucle es largo, Numba puede compensarlo.
        # Solo convierte los segmentos que se van a comprobar.
        if check_body_from_index < len(self.snake_body):
            body_to_check = self.snake_body[check_body_from_index:]
            snake_body_np = np.array([[seg['x'], seg['y']]
                                     for seg in body_to_check], dtype=np.int32)
            # El índice para la función numba es 0 sobre el array pasado
            return SnakeLogic._is_body_collision_numba(x, y, snake_body_np, 0)
        return False

    def _is_general_collision(self, x, y, check_body_from_index=0):
        if self._is_wall_collision(x, y):
            return True
        # Para _is_body_collision, x,y es la nueva cabeza.
        # Si check_body_from_index es 0, se comprueba contra todo el cuerpo actual.
        # Si check_body_from_index es 1, se comprueba contra el cuerpo excepto la cabeza actual (útil si la cabeza aún no se ha movido).
        if self._is_body_collision(x, y, check_body_from_index):
            return True
        return False

    def _is_immediate_trap(self, x_start, y_start, dx_forward, dy_forward, check_body_from_index_for_trap_check=0):
        # Esta función llama a _is_general_collision, que a su vez puede usar Numba.
        next_x_straight = x_start + dx_forward
        next_y_straight = y_start + dy_forward
        straight_blocked = self._is_general_collision(
            next_x_straight, next_y_straight, check_body_from_index_for_trap_check)

        dx_left_rel, dy_left_rel = -dy_forward, dx_forward
        next_x_left_rel, next_y_left_rel = x_start + dx_left_rel, y_start + dy_left_rel
        left_rel_blocked = self._is_general_collision(
            next_x_left_rel, next_y_left_rel, check_body_from_index_for_trap_check)

        dx_right_rel, dy_right_rel = dy_forward, -dx_forward
        next_x_right_rel, next_y_right_rel = x_start + \
            dx_right_rel, y_start + dy_right_rel
        right_rel_blocked = self._is_general_collision(
            next_x_right_rel, next_y_right_rel, check_body_from_index_for_trap_check)

        return straight_blocked and left_rel_blocked and right_rel_blocked

    def get_state(self):
        if not self.snake_body:  # Manejar caso raro de cuerpo vacío
            # Devuelve un estado por defecto o maneja el error apropiadamente
            # Asumiendo que el nuevo STATE_SIZE es 11
            default_state_values = (0,) * 11
            return default_state_values

        head = self.snake_body[0]
        current_dx_eff, current_dy_eff = self.snake_dx_step, self.snake_dy_step
        if current_dx_eff == 0 and current_dy_eff == 0:
            # Asumir movimiento a la derecha al inicio
            current_dx_eff, current_dy_eff = self.step_size, 0

        # Peligros (índice de chequeo de cuerpo = 1 porque la cabeza actual no cuenta como obstáculo para el *próximo* movimiento)
        peligro_recto = self._is_general_collision(
            head['x'] + current_dx_eff, head['y'] + current_dy_eff, 1)
        peligro_izquierda_rel = self._is_general_collision(
            head['x'] - current_dy_eff, head['y'] + current_dx_eff, 1)
        peligro_derecha_rel = self._is_general_collision(
            head['x'] + current_dy_eff, head['y'] - current_dx_eff, 1)

        # Callejones (índice de chequeo de cuerpo = 0 porque se simula un movimiento desde una *nueva* posición)
        callejon_recto = 0
        if not peligro_recto:
            callejon_recto = self._is_immediate_trap(
                head['x'] + current_dx_eff, head['y'] + current_dy_eff, current_dx_eff, current_dy_eff, 0)
        callejon_izquierda_rel = 0
        if not peligro_izquierda_rel:
            callejon_izquierda_rel = self._is_immediate_trap(
                head['x'] - current_dy_eff, head['y'] + current_dx_eff, -current_dy_eff, current_dx_eff, 0)
        callejon_derecha_rel = 0
        if not peligro_derecha_rel:
            callejon_derecha_rel = self._is_immediate_trap(
                head['x'] + current_dy_eff, head['y'] - current_dx_eff, current_dy_eff, -current_dx_eff, 0)

        food_dir_x = np.sign(self.food_x - head['x'])
        food_dir_y = np.sign(self.food_y - head['y'])

        dir_category = 4  # 4 para None/Initial
        if self.snake_dx_step == self.step_size:
            dir_category = 3  # RIGHT
        elif self.snake_dx_step == -self.step_size:
            dir_category = 2  # LEFT
        elif self.snake_dy_step == self.step_size:
            dir_category = 0  # UP
        elif self.snake_dy_step == -self.step_size:
            dir_category = 1  # DOWN

        # --- NUEVAS CARACTERÍSTICAS: Dirección a la cola ---
        tail_dir_x, tail_dir_y = 0, 0  # Valores por defecto si la serpiente es muy corta
        if len(self.snake_body) > 1:  # Necesita al menos cabeza y un segmento de cola
            tail_segment = self.snake_body[-1]
            tail_dir_x = np.sign(tail_segment['x'] - head['x'])
            tail_dir_y = np.sign(tail_segment['y'] - head['y'])

        state = (
            int(peligro_recto), int(peligro_izquierda_rel), int(
                peligro_derecha_rel),
            int(callejon_recto), int(callejon_izquierda_rel), int(
                callejon_derecha_rel),
            int(food_dir_x), int(food_dir_y),
            dir_category,
            int(tail_dir_x), int(tail_dir_y)  # Añadidas al estado
        )
        # STATE_SIZE ahora es 9 + 2 = 11
        return state

    def step(self, action_index):
        reward = REWARD_MOVE_LOGIC
        self.game_over = False
        info = {'collision_type': None, 'ate_food': False}

        potential_move = ACTION_MAP_LOGIC.get(action_index)
        current_dx, current_dy = self.snake_dx_step, self.snake_dy_step
        requested_dx, requested_dy = current_dx, current_dy

        if potential_move:
            is_opposite_move = False
            if not (current_dx == 0 and current_dy == 0):
                if (potential_move['dx'] == -current_dx and current_dx != 0) or \
                   (potential_move['dy'] == -current_dy and current_dy != 0):
                    is_opposite_move = True
            if not is_opposite_move:
                requested_dx, requested_dy = potential_move['dx'], potential_move['dy']

        self.snake_dx_step, self.snake_dy_step = requested_dx, requested_dy

        # Si no hay movimiento (ej. primer paso o acción no válida)
        if self.snake_dx_step == 0 and self.snake_dy_step == 0:
            return self.get_state(), reward, self.game_over, info

        head = self.snake_body[0]
        new_head_x = head['x'] + self.snake_dx_step
        new_head_y = head['y'] + self.snake_dy_step

        if self._is_wall_collision(new_head_x, new_head_y):
            self.game_over = True
            reward += REWARD_WALL_COLLISION_LOGIC
            info['collision_type'] = "wall"
        # Para _is_body_collision, check_body_from_index es 0 porque la nueva cabeza (new_head_x, new_head_y)
        # aún no se ha añadido a self.snake_body, por lo que debe comprobarse contra todos los segmentos actuales.
        elif self._is_body_collision(new_head_x, new_head_y, 0):
            self.game_over = True
            reward += REWARD_SELF_COLLISION_LOGIC
            info['collision_type'] = "self"

        if self.game_over:
            return self.get_state(), reward, self.game_over, info

        new_head_segment = {'x': new_head_x, 'y': new_head_y}
        ate_food = False
        if new_head_x == self.food_x and new_head_y == self.food_y:
            ate_food = True
            self.score += 1
            reward += REWARD_FOOD_LOGIC
            info['ate_food'] = True

        self.snake_body.insert(0, new_head_segment)

        if not ate_food:
            self.snake_body.pop()
        else:
            self._place_food()

        return self.get_state(), reward, self.game_over, info
