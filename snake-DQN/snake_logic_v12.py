# snake_logic_v12.py
import random
import numpy as np
from numba import jit, int32
from numba.typed import List as NumbaList
# NumbaDict y NumbaTypes ya no son necesarios para el BFS con array de visitados
# from numba.typed import Dict as NumbaDict
# from numba.core import types as NumbaTypes

# --- Constantes del Juego ---
SCREEN_WIDTH_LOGIC = 250
SCREEN_HEIGHT_LOGIC = 250
STEP_SIZE_LOGIC = 20

ACTION_MAP_LOGIC = {
    0: {'dx': 0, 'dy': STEP_SIZE_LOGIC},    # UP (Y aumenta en lógica)
    1: {'dx': 0, 'dy': -STEP_SIZE_LOGIC},   # DOWN (Y disminuye en lógica)
    2: {'dx': -STEP_SIZE_LOGIC, 'dy': 0},   # LEFT
    3: {'dx': STEP_SIZE_LOGIC, 'dy': 0}     # RIGHT
}

# --- Recompensas Actualizadas en Revisión ---
REWARD_FOOD_LOGIC = 5
REWARD_WALL_COLLISION_LOGIC = -5
REWARD_SELF_COLLISION_LOGIC = -5
REWARD_MOVE_LOGIC = 0

# --- Funciones Auxiliares JITeadas ---


@jit(nopython=True, cache=True)
def _is_wall_collision_numba_static(x: int, y: int, width: int, height: int, step_size: int) -> bool:
    half_step = step_size // 2
    if not (half_step <= x < width - half_step + step_size % 2 and
            half_step <= y < height - half_step + step_size % 2):
        return True
    return False


@jit(nopython=True, cache=True)
def _is_body_collision_numba_static(x: int, y: int, snake_body_np: np.ndarray) -> bool:
    for i in range(snake_body_np.shape[0]):
        if x == snake_body_np[i, 0] and y == snake_body_np[i, 1]:
            return True
    return False


@jit(nopython=True, cache=True)
def _coords_to_indices_numba(x: int, y: int, step_size: int):
    """Convierte coordenadas del juego (centro de celda) a índices de array."""
    ix = (x - (step_size // 2)) // step_size
    iy = (y - (step_size // 2)) // step_size
    return ix, iy


@jit(nopython=True, cache=True)
def _bfs_path_exists_numba(
    start_node_x: int, start_node_y: int,
    end_node_x: int, end_node_y: int,
    obstacles_np: np.ndarray,
    grid_width: int, grid_height: int, step_size: int
) -> bool:

    _start_node_x = np.int32(start_node_x)
    _start_node_y = np.int32(start_node_y)
    _end_node_x = np.int32(end_node_x)
    _end_node_y = np.int32(end_node_y)

    q = NumbaList()
    q.append((_start_node_x, _start_node_y))

    # Dimensiones de la grilla para el array de visitados
    max_ix = (grid_width - step_size // 2 - (step_size % 2)) // step_size
    max_iy = (grid_height - step_size // 2 - (step_size % 2)) // step_size

    # Array 2D para visitados. Asegúrate de que los índices no se salgan.
    # El tamaño es max_ix + 1 porque los índices van de 0 a max_ix.
    visited_arr = np.zeros((max_ix + 1, max_iy + 1), dtype=np.bool_)

    start_ix, start_iy = _coords_to_indices_numba(
        _start_node_x, _start_node_y, step_size)
    if not (0 <= start_ix <= max_ix and 0 <= start_iy <= max_iy):  # Comprobación de seguridad
        return False  # Nodo inicial fuera de la grilla de índices
    visited_arr[start_ix, start_iy] = True

    possible_moves_dx = np.array([0, 0, -step_size, step_size], dtype=np.int32)
    possible_moves_dy = np.array([step_size, -step_size, 0, 0], dtype=np.int32)

    while len(q) > 0:
        current_x, current_y = q.pop(0)

        if current_x == _end_node_x and current_y == _end_node_y:
            return True

        for i in range(len(possible_moves_dx)):
            dx_move = possible_moves_dx[i]
            dy_move = possible_moves_dy[i]

            next_x = np.int32(current_x + dx_move)
            next_y = np.int32(current_y + dy_move)

            next_ix, next_iy = _coords_to_indices_numba(
                next_x, next_y, step_size)

            # Comprobar límites del array de visitados
            if not (0 <= next_ix <= max_ix and 0 <= next_iy <= max_iy):
                continue  # Índice fuera de los límites del array

            if not visited_arr[next_ix, next_iy]:
                if not _is_wall_collision_numba_static(next_x, next_y, grid_width, grid_height, step_size):
                    is_obstacle = False
                    if obstacles_np.shape[0] > 0:
                        for j in range(obstacles_np.shape[0]):
                            if next_x == obstacles_np[j, 0] and next_y == obstacles_np[j, 1]:
                                is_obstacle = True
                                break

                    if not is_obstacle or (next_x == _end_node_x and next_y == _end_node_y):
                        visited_arr[next_ix, next_iy] = True
                        q.append((next_x, next_y))
    return False


class SnakeLogic:
    def __init__(self, width, height, step_size):
        self.width = width
        self.height = height
        self.step_size = step_size
        self.snake_body = []
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

    def _is_wall_collision(self, x, y):
        return _is_wall_collision_numba_static(x, y, self.width, self.height, self.step_size)

    def _is_body_collision(self, x, y, check_body_from_index=0):
        if not self.snake_body:
            return False

        if check_body_from_index < len(self.snake_body):
            body_to_check_list = self.snake_body[check_body_from_index:]
            if not body_to_check_list:
                return False
            snake_body_np = np.array([[seg['x'], seg['y']]
                                     for seg in body_to_check_list], dtype=np.int32)
            return _is_body_collision_numba_static(x, y, snake_body_np)
        return False

    def _is_general_collision(self, x, y, check_body_from_index=0):
        if self._is_wall_collision(x, y):
            return True
        if self._is_body_collision(x, y, check_body_from_index):
            return True
        return False

    def _is_immediate_trap(self, x_start, y_start, dx_forward, dy_forward, check_body_from_index_for_trap_check=0):
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

    def _calculate_tail_accessibility_after_eating(self):
        if not self.snake_body:
            return 0

        sim_head_pos_dict = {'x': self.food_x, 'y': self.food_y}
        sim_snake_body_list_of_dicts = [
            sim_head_pos_dict] + list(self.snake_body)

        if len(sim_snake_body_list_of_dicts) < 2:
            return 1

        sim_tail_pos_dict = sim_snake_body_list_of_dicts[-1]

        obstacles_for_bfs_py_list = []
        if len(sim_snake_body_list_of_dicts) > 2:
            for segment_dict in sim_snake_body_list_of_dicts[1:-1]:
                obstacles_for_bfs_py_list.append(
                    [np.int32(segment_dict['x']), np.int32(segment_dict['y'])])

        if obstacles_for_bfs_py_list:
            obstacles_np = np.array(obstacles_for_bfs_py_list, dtype=np.int32)
        else:
            obstacles_np = np.empty((0, 2), dtype=np.int32)

        path_exists = _bfs_path_exists_numba(
            start_node_x=sim_head_pos_dict['x'],
            start_node_y=sim_head_pos_dict['y'],
            end_node_x=sim_tail_pos_dict['x'],
            end_node_y=sim_tail_pos_dict['y'],
            obstacles_np=obstacles_np,
            grid_width=self.width,
            grid_height=self.height,
            step_size=self.step_size
        )
        return 1 if path_exists else 0

    def get_state(self):
        if not self.snake_body:
            return (0,) * 12

        head = self.snake_body[0]
        current_dx_eff, current_dy_eff = self.snake_dx_step, self.snake_dy_step
        if current_dx_eff == 0 and current_dy_eff == 0:
            if len(self.snake_body) > 1:
                prev_segment = self.snake_body[1]
                if prev_segment['x'] < head['x']:
                    current_dx_eff = self.step_size
                elif prev_segment['x'] > head['x']:
                    current_dx_eff = -self.step_size
                elif prev_segment['y'] < head['y']:
                    current_dy_eff = self.step_size
                elif prev_segment['y'] > head['y']:
                    current_dy_eff = -self.step_size
                else:
                    current_dx_eff = self.step_size
            else:
                current_dx_eff = self.step_size

        dx_fwd, dy_fwd = current_dx_eff, current_dy_eff
        dx_left, dy_left = -dy_fwd, dx_fwd
        dx_right, dy_right = dy_fwd, -dx_fwd

        peligro_recto = self._is_general_collision(
            head['x'] + dx_fwd, head['y'] + dy_fwd, 1)
        peligro_izquierda_rel = self._is_general_collision(
            head['x'] + dx_left, head['y'] + dy_left, 1)
        peligro_derecha_rel = self._is_general_collision(
            head['x'] + dx_right, head['y'] + dy_right, 1)

        callejon_recto = 0
        if not peligro_recto:
            callejon_recto = self._is_immediate_trap(
                head['x'] + dx_fwd, head['y'] + dy_fwd,
                dx_fwd, dy_fwd, 0)
        callejon_izquierda_rel = 0
        if not peligro_izquierda_rel:
            callejon_izquierda_rel = self._is_immediate_trap(
                head['x'] + dx_left, head['y'] + dy_left,
                dx_left, dy_left, 0)
        callejon_derecha_rel = 0
        if not peligro_derecha_rel:
            callejon_derecha_rel = self._is_immediate_trap(
                head['x'] + dx_right, head['y'] + dy_right,
                dx_right, dy_right, 0)

        food_dir_x = np.sign(self.food_x - head['x'])
        food_dir_y = np.sign(self.food_y - head['y'])

        dir_category = 4
        if self.snake_dx_step == self.step_size:
            dir_category = 3
        elif self.snake_dx_step == -self.step_size:
            dir_category = 2
        elif self.snake_dy_step == self.step_size:
            dir_category = 0
        elif self.snake_dy_step == -self.step_size:
            dir_category = 1

        tail_dir_x, tail_dir_y = 0, 0
        if len(self.snake_body) > 1:
            tail_segment = self.snake_body[-1]
            tail_dir_x = np.sign(tail_segment['x'] - head['x'])
            tail_dir_y = np.sign(tail_segment['y'] - head['y'])

        tail_accessible_after_eating = self._calculate_tail_accessibility_after_eating()

        state = (
            int(peligro_recto), int(peligro_izquierda_rel), int(
                peligro_derecha_rel),
            int(callejon_recto), int(callejon_izquierda_rel), int(
                callejon_derecha_rel),
            int(food_dir_x), int(food_dir_y),
            dir_category,
            int(tail_dir_x), int(tail_dir_y),
            int(tail_accessible_after_eating)
        )
        return state

    def step(self, action_index):
        reward = REWARD_MOVE_LOGIC
        self.game_over = False
        info = {'collision_type': None, 'ate_food': False}

        potential_move = ACTION_MAP_LOGIC.get(action_index)
        current_dx_actual, current_dy_actual = self.snake_dx_step, self.snake_dy_step
        requested_dx, requested_dy = current_dx_actual, current_dy_actual

        if potential_move:
            is_opposite_move = False
            if not (current_dx_actual == 0 and current_dy_actual == 0):
                if (potential_move['dx'] == -current_dx_actual and current_dx_actual != 0) or \
                   (potential_move['dy'] == -current_dy_actual and current_dy_actual != 0):
                    is_opposite_move = True
            if not is_opposite_move:
                requested_dx, requested_dy = potential_move['dx'], potential_move['dy']

        self.snake_dx_step, self.snake_dy_step = requested_dx, requested_dy

        if self.snake_dx_step == 0 and self.snake_dy_step == 0:
            if len(self.snake_body) > 0:
                head_for_default = self.snake_body[0]
                if len(self.snake_body) > 1:
                    prev_segment = self.snake_body[1]
                    if prev_segment['x'] < head_for_default['x']:
                        self.snake_dx_step = self.step_size
                    elif prev_segment['x'] > head_for_default['x']:
                        self.snake_dx_step = -self.step_size
                    elif prev_segment['y'] < head_for_default['y']:
                        self.snake_dy_step = self.step_size
                    elif prev_segment['y'] > head_for_default['y']:
                        self.snake_dy_step = -self.step_size
                    else:
                        self.snake_dx_step = self.step_size
                elif len(self.snake_body) == 1:
                    self.snake_dx_step = self.step_size

            if self.snake_dx_step == 0 and self.snake_dy_step == 0:
                return self.get_state(), reward, self.game_over, info

        head = self.snake_body[0]
        new_head_x = head['x'] + self.snake_dx_step
        new_head_y = head['y'] + self.snake_dy_step

        if self._is_wall_collision(new_head_x, new_head_y):
            self.game_over = True
            reward += REWARD_WALL_COLLISION_LOGIC
            info['collision_type'] = "wall"
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
