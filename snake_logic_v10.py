# snake_logic_v10.py
import random
import numpy as np

# --- Constantes del Juego (Se pueden pasar al constructor si quieres flexibilidad) ---
SCREEN_WIDTH_LOGIC = 250  # Usar un sufijo para distinguirlas si es necesario
SCREEN_HEIGHT_LOGIC = 250
STEP_SIZE_LOGIC = 20
# No necesitamos POINTS_PER_FOOD aquí, la puntuación es parte de la lógica del juego en sí

# --- Definiciones para la IA ---
ACTION_MAP_LOGIC = {
    0: {'dx': 0, 'dy': STEP_SIZE_LOGIC},    # UP
    1: {'dx': 0, 'dy': -STEP_SIZE_LOGIC},   # DOWN
    2: {'dx': -STEP_SIZE_LOGIC, 'dy': 0},   # LEFT
    3: {'dx': STEP_SIZE_LOGIC, 'dy': 0}     # RIGHT
}

# --- Recompensas ---
# Puedes definir estas aquí o pasarlas/obtenerlas del trainer
REWARD_FOOD_LOGIC = 20
REWARD_WALL_COLLISION_LOGIC = -50
REWARD_SELF_COLLISION_LOGIC = -100  # O el valor que decidas
REWARD_MOVE_LOGIC = -1  # O tu valor de reward shaping
# REWARD_STUCK_LOOP se maneja en el trainer basado en info de 'ate_food' y steps


class SnakeLogic:
    def __init__(self, width, height, step_size):
        self.width = width
        self.height = height
        self.step_size = step_size

        self.snake_body = []  # Lista de diccionarios {'x': x_val, 'y': y_val}
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self.food_x = 0
        self.food_y = 0
        self.score = 0
        self.game_over = False
        # No necesitamos last_collision_type aquí a menos que el estado lo requiera explícitamente
        # Si el estado lo necesita, entonces sí. Tu estado actual no parece usarlo.

        self.setup()  # Configuración inicial

    def setup(self):
        self.game_over = False
        self.score = 0

        # Centrar la serpiente, asegurando que esté alineada con la cuadrícula
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
        self.snake_dx_step = 0  # Movimiento inicial nulo, el agente debe decidir
        self.snake_dy_step = 0
        self._place_food()
        return self.get_state()  # Devuelve el estado inicial

    def reset(self):
        return self.setup()

    def _place_food(self):
        placed_on_snake = True
        while placed_on_snake:
            # Asegurar que la comida también esté alineada con la cuadrícula y centrada en su celda
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
        half_step = self.step_size // 2
        if not (half_step <= x < self.width - half_step + self.step_size % 2 and  # Ajuste para bordes
                half_step <= y < self.height - half_step + self.step_size % 2):
            return True
        return False

    def _is_body_collision(self, x, y, check_body_from_index=0):
        for i in range(check_body_from_index, len(self.snake_body)):
            segment = self.snake_body[i]
            # Comparamos centros exactos para colisión en cuadrícula
            if x == segment['x'] and y == segment['y']:
                return True
        return False

    def _is_general_collision(self, x, y, check_body_from_index=0):
        if self._is_wall_collision(x, y):
            return True
        if self._is_body_collision(x, y, check_body_from_index):
            return True
        return False

    def _is_immediate_trap(self, x_start, y_start, dx_forward, dy_forward, check_body_from_index_for_trap_check=0):
        # Esta función asume que (x_start, y_start) es una posición válida
        # y que (dx_forward, dy_forward) es la dirección actual de movimiento desde allí.

        # Chequear adelante
        next_x_straight = x_start + dx_forward
        next_y_straight = y_start + dy_forward
        straight_blocked = self._is_general_collision(
            next_x_straight, next_y_straight, check_body_from_index_for_trap_check)

        # Chequear izquierda relativa
        dx_left_rel = -dy_forward
        dy_left_rel = dx_forward
        next_x_left_rel = x_start + dx_left_rel
        next_y_left_rel = y_start + dy_left_rel
        left_rel_blocked = self._is_general_collision(
            next_x_left_rel, next_y_left_rel, check_body_from_index_for_trap_check)

        # Chequear derecha relativa
        dx_right_rel = dy_forward
        dy_right_rel = -dx_forward
        next_x_right_rel = x_start + dx_right_rel
        next_y_right_rel = y_start + dy_right_rel
        right_rel_blocked = self._is_general_collision(
            next_x_right_rel, next_y_right_rel, check_body_from_index_for_trap_check)

        return straight_blocked and left_rel_blocked and right_rel_blocked

    def get_state(self):
        head = self.snake_body[0]

        # Determinar dirección actual efectiva para 'peligro_recto'
        # Si la serpiente está quieta al inicio, asumimos que se movería a la derecha
        current_dx_eff, current_dy_eff = self.snake_dx_step, self.snake_dy_step
        if current_dx_eff == 0 and current_dy_eff == 0:  # Solo al inicio del juego
            current_dx_eff = self.step_size
            current_dy_eff = 0

        # Peligro recto
        check_x_straight = head['x'] + current_dx_eff
        check_y_straight = head['y'] + current_dy_eff
        # Para peligro_recto, la colisión con el cuerpo se chequearía a partir del segundo segmento (índice 1)
        # porque la cabeza aún no se ha movido.
        peligro_recto = self._is_general_collision(
            check_x_straight, check_y_straight, 1)

        # Peligro izquierda relativa
        dx_left = -current_dy_eff
        dy_left = current_dx_eff
        check_x_left = head['x'] + dx_left
        check_y_left = head['y'] + dy_left
        peligro_izquierda_rel = self._is_general_collision(
            check_x_left, check_y_left, 1)

        # Peligro derecha relativa
        dx_right = current_dy_eff
        dy_right = -current_dx_eff
        check_x_right = head['x'] + dx_right
        check_y_right = head['y'] + dy_right
        peligro_derecha_rel = self._is_general_collision(
            check_x_right, check_y_right, 1)

        # Callejones (usando la nueva posición potencial como inicio para el chequeo)
        # Nota: El check_body_from_index para _is_immediate_trap debería ser 0 si la cabeza ya
        # se ha movido a esa posición antes de llamar a _is_immediate_trap.
        # Aquí, estamos evaluando desde la posición actual de la cabeza.
        callejon_recto = 0
        if not peligro_recto:  # Si no hay un peligro inmediato al frente...
            # ...evaluamos si moverse al frente nos mete en un callejón.
            # El 'check_body_from_index_for_trap_check' debe ser 0 porque _is_immediate_trap
            # simula movimientos desde (check_x_straight, check_y_straight)
            # y la cabeza aún no está allí en self.snake_body.
            callejon_recto = self._is_immediate_trap(
                check_x_straight, check_y_straight, current_dx_eff, current_dy_eff, 0)

        callejon_izquierda_rel = 0
        if not peligro_izquierda_rel:
            callejon_izquierda_rel = self._is_immediate_trap(
                check_x_left, check_y_left, dx_left, dy_left, 0)

        callejon_derecha_rel = 0
        if not peligro_derecha_rel:
            callejon_derecha_rel = self._is_immediate_trap(
                check_x_right, check_y_right, dx_right, dy_right, 0)

        # Dirección de la comida
        food_dir_x = np.sign(self.food_x - head['x'])
        food_dir_y = np.sign(self.food_y - head['y'])

        # Dirección actual de la serpiente
        dir_category = 4  # 4 para None/Initial
        if self.snake_dx_step == self.step_size:
            dir_category = 3  # RIGHT
        elif self.snake_dx_step == -self.step_size:
            dir_category = 2  # LEFT
        elif self.snake_dy_step == self.step_size:
            dir_category = 0  # UP
        elif self.snake_dy_step == -self.step_size:
            dir_category = 1  # DOWN

        state = (
            int(peligro_recto), int(peligro_izquierda_rel), int(
                peligro_derecha_rel),
            int(callejon_recto), int(callejon_izquierda_rel), int(
                callejon_derecha_rel),
            int(food_dir_x), int(food_dir_y),
            dir_category
        )
        return state

    def step(self, action_index):  # action_index es 0, 1, 2, o 3
        reward = REWARD_MOVE_LOGIC
        self.game_over = False  # Asumimos que no termina a menos que una colisión ocurra
        info = {'collision_type': None, 'ate_food': False}

        potential_move = ACTION_MAP_LOGIC.get(action_index)

        # Direccion actual antes de aplicar la nueva acción
        current_dx = self.snake_dx_step
        current_dy = self.snake_dy_step

        requested_dx = current_dx  # Mantener dirección si la acción no es válida o es opuesta
        requested_dy = current_dy

        if potential_move:
            # Evitar que la serpiente se mueva en la dirección opuesta inmediata
            # Solo aplica si la serpiente ya se está moviendo
            is_opposite_move = False
            if not (current_dx == 0 and current_dy == 0):  # Si no es el primer movimiento
                if potential_move['dx'] == -current_dx and current_dx != 0:
                    is_opposite_move = True
                if potential_move['dy'] == -current_dy and current_dy != 0:
                    is_opposite_move = True

            if not is_opposite_move:
                requested_dx = potential_move['dx']
                requested_dy = potential_move['dy']

        self.snake_dx_step = requested_dx
        self.snake_dy_step = requested_dy

        # Si la serpiente está quieta (solo al inicio antes del primer movimiento válido)
        # o si la acción resultante es no moverse (no debería pasar con el fix de arriba),
        # no hacemos nada más que devolver el estado actual.
        if self.snake_dx_step == 0 and self.snake_dy_step == 0:
            # Esto solo debería ocurrir si es el primer paso y la acción es inválida
            # o si se intenta forzar un no-movimiento.
            # Para el primer paso, el trainer debe dar una acción válida.
            # Si es el primer movimiento, la cabeza no se mueve, la recompensa es REWARD_MOVE.
            return self.get_state(), reward, self.game_over, info

        head = self.snake_body[0]
        new_head_x = head['x'] + self.snake_dx_step
        new_head_y = head['y'] + self.snake_dy_step

        # 1. Comprobar colisión con paredes
        if self._is_wall_collision(new_head_x, new_head_y):
            self.game_over = True
            reward += REWARD_WALL_COLLISION_LOGIC
            info['collision_type'] = "wall"
        # 2. Comprobar colisión con el propio cuerpo (si no chocó con la pared)
        # Para la nueva cabeza, check_body_from_index debe ser 0 porque la nueva cabeza
        # aún no forma parte de self.snake_body.
        elif self._is_body_collision(new_head_x, new_head_y, 0):
            self.game_over = True
            reward += REWARD_SELF_COLLISION_LOGIC
            info['collision_type'] = "self"

        # Si el juego terminó, devolver estado, recompensa, done, info
        if self.game_over:
            return self.get_state(), reward, self.game_over, info

        # Si no hay colisión, mover la serpiente
        new_head_segment = {'x': new_head_x, 'y': new_head_y}

        ate_food = False
        # Comprobar si se come la comida (comparando centros)
        if new_head_x == self.food_x and new_head_y == self.food_y:
            ate_food = True
            self.score += 1  # O usa POINTS_PER_FOOD_LOGIC si lo defines
            reward += REWARD_FOOD_LOGIC
            info['ate_food'] = True

        self.snake_body.insert(0, new_head_segment)

        if not ate_food:
            self.snake_body.pop()  # Quitar el último segmento si no comió
        else:
            self._place_food()  # Colocar nueva comida
            # Comprobar si ha ganado llenando la pantalla (opcional)
            # if len(self.snake_body) >= (self.width // self.step_size) * (self.height // self.step_size):
            #     self.game_over = True # Considerar como game over para terminar episodio

        next_observation = self.get_state()
        return next_observation, reward, self.game_over, info
