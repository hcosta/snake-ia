# snake_v9.py

# Se ha separado el game over en dos partes (chocar con pared y chocar consigo misma)
# y se ha añadido un nuevo tipo de colisión: callejón. Además se han añadido
# recompensas específicas para cada tipo de colisión.

import arcade
import random
import numpy as np

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 250
SCREEN_HEIGHT = 250
SCREEN_TITLE = "Snake IA v9"

# --- Constantes del Juego ---
SNAKE_SIZE = 20
STEP_SIZE = SNAKE_SIZE
FOOD_SIZE = SNAKE_SIZE
POINTS_PER_FOOD = 10

# --- Colores ---
BACKGROUND_COLOR = arcade.color.AMAZON
SNAKE_COLOR = arcade.color.RED_DEVIL
FOOD_COLOR = arcade.color.APPLE_GREEN
GAMEOVER_COLOR = arcade.color.BLACK_LEATHER_JACKET
SCORE_COLOR = arcade.color.BLACK

# --- Definiciones para la IA ---
ACTION_MAP = {
    0: {'dx': 0, 'dy': STEP_SIZE},    # UP
    1: {'dx': 0, 'dy': -STEP_SIZE},   # DOWN
    2: {'dx': -STEP_SIZE, 'dy': 0},   # LEFT
    3: {'dx': STEP_SIZE, 'dy': 0}     # RIGHT
}

# --- Recompensas (NUEVO y MODIFICADO) ---
REWARD_FOOD = 10
# REWARD_GAME_OVER = -100 # Eliminada, ahora usamos recompensas específicas
REWARD_WALL_COLLISION = -50  # Penalización por chocar con la pared
REWARD_SELF_COLLISION = -100  # Penalización (mayor) por chocar consigo misma
# Pequeña penalización por cada movimiento para incentivar la eficiencia
REWARD_MOVE = -0.1
# Penalización por quedarse atascado o en bucles ineficientes
REWARD_STUCK_LOOP = -50


class SnakeGame(arcade.Window):
    def __init__(self, width, height, title, ai_controlled=False, time_per_move_human=0.1):
        super().__init__(width, height, title, update_rate=1/60)
        arcade.set_background_color(BACKGROUND_COLOR)

        self.ai_controlled = ai_controlled
        self.movement_timer = 0.0
        self.time_per_move = time_per_move_human

        self.game_over = False
        self.score = 0
        self.snake_body = []
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self.food_x = 0
        self.food_y = 0

        self.training_should_stop = False
        self.last_collision_type = None  # NUEVO: Para registrar el tipo de colisión
        self.setup()

    def setup(self):
        self.game_over = False
        self.score = 0
        head_start_x = (SCREEN_WIDTH // 2 // STEP_SIZE) * \
            STEP_SIZE + (STEP_SIZE // 2)
        head_start_y = (SCREEN_HEIGHT // 2 // STEP_SIZE) * \
            STEP_SIZE + (STEP_SIZE // 2)
        self.snake_body = [
            {'x': head_start_x, 'y': head_start_y},
            {'x': head_start_x - STEP_SIZE, 'y': head_start_y},
            {'x': head_start_x - 2 * STEP_SIZE, 'y': head_start_y}
        ]
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self._place_food()
        self.training_should_stop = False
        self.last_collision_type = None  # Reiniciar tipo de colisión
        return self.get_state()

    def reset(self):
        return self.setup()

    def _place_food(self):
        placed_on_snake = True
        while placed_on_snake:
            self.food_x = random.randrange(
                STEP_SIZE // 2, SCREEN_WIDTH - STEP_SIZE // 2 + 1, STEP_SIZE)
            self.food_y = random.randrange(
                STEP_SIZE // 2, SCREEN_HEIGHT - STEP_SIZE // 2 + 1, STEP_SIZE)
            placed_on_snake = False
            for segment in self.snake_body:
                if abs(segment['x'] - self.food_x) < STEP_SIZE // 2 and \
                   abs(segment['y'] - self.food_y) < STEP_SIZE // 2:
                    placed_on_snake = True
                    break

    # --- NUEVAS FUNCIONES DE DETECCIÓN DE COLISIÓN ESPECÍFICAS ---
    def _is_wall_collision(self, x, y):
        """Comprueba si las coordenadas (x, y) están fuera de los límites."""
        x_coord_min = STEP_SIZE // 2
        y_coord_min = STEP_SIZE // 2
        x_coord_max_center = SCREEN_WIDTH - (STEP_SIZE // 2)
        y_coord_max_center = SCREEN_HEIGHT - (STEP_SIZE // 2)

        if not (x_coord_min <= x <= x_coord_max_center and
                y_coord_min <= y <= y_coord_max_center):
            return True
        return False

    def _is_body_collision(self, x, y, check_body_from_index=0):
        """Comprueba si las coordenadas (x, y) colisionan con el cuerpo de la serpiente."""
        # `check_body_from_index` se usa para evitar chequear la cabeza contra sí misma
        # si la cabeza ya se ha movido a la nueva posición antes de llamar a esta función.
        for i in range(check_body_from_index, len(self.snake_body)):
            segment = self.snake_body[i]
            if abs(x - segment['x']) < STEP_SIZE // 2 and \
               abs(y - segment['y']) < STEP_SIZE // 2:
                return True
        return False

    def _is_general_collision(self, x, y, check_body_from_index=0):
        """
        Comprueba colisión con bordes O con el cuerpo.
        Utiliza las funciones específicas _is_wall_collision y _is_body_collision.
        Esta función es usada principalmente por get_state para determinar peligros.
        """
        if self._is_wall_collision(x, y):
            return True
        if self._is_body_collision(x, y, check_body_from_index):
            return True
        return False
    # --- FIN DE NUEVAS FUNCIONES DE DETECCIÓN DE COLISIÓN ---

    def _is_immediate_trap(self, x_start, y_start, dx_forward, dy_forward, check_body_from_index_for_trap_check=0):
        next_x_straight = x_start + dx_forward
        next_y_straight = y_start + dy_forward
        straight_blocked = self._is_general_collision(  # Usa la función general
            next_x_straight, next_y_straight, check_body_from_index_for_trap_check)

        dx_left_rel = -dy_forward
        dy_left_rel = dx_forward
        next_x_left_rel = x_start + dx_left_rel
        next_y_left_rel = y_start + dy_left_rel
        left_rel_blocked = self._is_general_collision(  # Usa la función general
            next_x_left_rel, next_y_left_rel, check_body_from_index_for_trap_check)

        dx_right_rel = dy_forward
        dy_right_rel = -dx_forward
        next_x_right_rel = x_start + dx_right_rel
        next_y_right_rel = y_start + dy_right_rel
        right_rel_blocked = self._is_general_collision(  # Usa la función general
            next_x_right_rel, next_y_right_rel, check_body_from_index_for_trap_check)

        return straight_blocked and left_rel_blocked and right_rel_blocked

    def get_state(self):
        head = self.snake_body[0]
        current_dx_eff, current_dy_eff = self.snake_dx_step, self.snake_dy_step
        if current_dx_eff == 0 and current_dy_eff == 0:
            current_dx_eff = STEP_SIZE
            current_dy_eff = 0

        check_x_straight = head['x'] + current_dx_eff
        check_y_straight = head['y'] + current_dy_eff
        peligro_recto = self._is_general_collision(  # Sigue usando _is_general_collision
            check_x_straight, check_y_straight, 1)

        dx_left = -current_dy_eff
        dy_left = current_dx_eff
        check_x_left = head['x'] + dx_left
        check_y_left = head['y'] + dy_left
        peligro_izquierda_rel = self._is_general_collision(  # Sigue usando _is_general_collision
            check_x_left, check_y_left, 1)

        dx_right = current_dy_eff
        dy_right = -current_dx_eff
        check_x_right = head['x'] + dx_right
        check_y_right = head['y'] + dy_right
        peligro_derecha_rel = self._is_general_collision(  # Sigue usando _is_general_collision
            check_x_right, check_y_right, 1)

        callejon_recto = 0
        if not peligro_recto:
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

        food_dir_x = np.sign(self.food_x - head['x'])
        food_dir_y = np.sign(self.food_y - head['y'])

        dir_category = 4
        if self.snake_dx_step == STEP_SIZE:
            dir_category = 3
        elif self.snake_dx_step == -STEP_SIZE:
            dir_category = 2
        elif self.snake_dy_step == STEP_SIZE:
            dir_category = 0
        elif self.snake_dy_step == -STEP_SIZE:
            dir_category = 1

        state = (
            int(peligro_recto), int(peligro_izquierda_rel), int(
                peligro_derecha_rel),
            int(callejon_recto), int(callejon_izquierda_rel), int(
                callejon_derecha_rel),
            int(food_dir_x), int(food_dir_y),
            dir_category
        )
        return state

    # --- FUNCIÓN step() MODIFICADA SIGNIFICATIVAMENTE ---
    def step(self, action):
        reward = REWARD_MOVE  # Recompensa base por movimiento
        done = False
        info = {'collision_type': None}  # Para información de depuración

        potential_move = ACTION_MAP.get(action)
        requested_dx = self.snake_dx_step
        requested_dy = self.snake_dy_step

        if potential_move:
            is_opposite_move = False
            if self.snake_dx_step != 0 or self.snake_dy_step != 0:
                if potential_move['dx'] == -self.snake_dx_step and self.snake_dx_step != 0:
                    is_opposite_move = True
                if potential_move['dy'] == -self.snake_dy_step and self.snake_dy_step != 0:
                    is_opposite_move = True
            if not is_opposite_move:
                requested_dx = potential_move['dx']
                requested_dy = potential_move['dy']

        self.snake_dx_step = requested_dx
        self.snake_dy_step = requested_dy

        if self.snake_dx_step == 0 and self.snake_dy_step == 0:
            current_observation = self.get_state()
            return current_observation, reward, done, info

        head = self.snake_body[0]
        new_head_x = head['x'] + self.snake_dx_step
        new_head_y = head['y'] + self.snake_dy_step

        # Comprobar colisiones específicas y asignar recompensas diferenciadas
        # check_body_from_index=0 para la nueva cabeza, ya que el cuerpo aún no se ha actualizado con ella.
        if self._is_wall_collision(new_head_x, new_head_y):
            done = True
            reward += REWARD_WALL_COLLISION  # Recompensa específica por chocar con la pared
            self.game_over = True
            info['collision_type'] = "wall"
            self.last_collision_type = "wall"  # Guardar tipo de colisión

        # Solo comprobar colisión con el cuerpo si no ha chocado ya con la pared
        elif self._is_body_collision(new_head_x, new_head_y, 0):
            done = True
            reward += REWARD_SELF_COLLISION  # Recompensa específica por chocar consigo misma
            self.game_over = True
            info['collision_type'] = "self"
            self.last_collision_type = "self"  # Guardar tipo de colisión

        if self.game_over:  # Si hay cualquier tipo de colisión, terminar el episodio
            return self.get_state(), reward, done, info

        # Si no hay colisión, mover la serpiente
        new_head_segment = {'x': new_head_x, 'y': new_head_y}
        ate_food = False
        if abs(new_head_x - self.food_x) < (STEP_SIZE // 2) and \
           abs(new_head_y - self.food_y) < (STEP_SIZE // 2):
            ate_food = True
            self.score += POINTS_PER_FOOD
            reward += REWARD_FOOD

        self.snake_body.insert(0, new_head_segment)

        if not ate_food:
            self.snake_body.pop()
        else:
            self._place_food()
            if len(self.snake_body) >= (SCREEN_WIDTH / STEP_SIZE) * (SCREEN_HEIGHT / STEP_SIZE):
                done = True
                self.game_over = True  # Considerar como game over para terminar episodio

        next_observation = self.get_state()
        return next_observation, reward, done, info
    # --- FIN DE LA FUNCIÓN step() MODIFICADA ---

    def draw_game_over(self):
        arcade.draw_text("GAME OVER", SCREEN_WIDTH / 2, (SCREEN_HEIGHT / 2) + 50, GAMEOVER_COLOR,
                         font_size=20, anchor_x="center", anchor_y="center")

        collision_msg = ""
        if self.last_collision_type == "wall":
            collision_msg = "Hit a wall!"
        elif self.last_collision_type == "self":
            collision_msg = "Hit itself!"

        arcade.draw_text(f"{self.score} puntos", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2,
                         GAMEOVER_COLOR, font_size=20, anchor_x="center", anchor_y="center")
        if collision_msg:
            arcade.draw_text(collision_msg, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 25,
                             GAMEOVER_COLOR, font_size=14, anchor_x="center", anchor_y="center")

        if not self.ai_controlled:
            arcade.draw_text("RESET = SPACE", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50,
                             GAMEOVER_COLOR, font_size=20, anchor_x="center", anchor_y="center")

    def draw_score(self):
        score_text = f"{self.score}"
        arcade.draw_text(score_text, 10, SCREEN_HEIGHT -
                         30, SCORE_COLOR, font_size=20)

    def on_draw(self):
        self.clear()
        if self.game_over:
            self.draw_game_over()  # draw_game_over ahora puede usar self.last_collision_type
            return
        arcade.draw_rectangle_filled(center_x=self.food_x, center_y=self.food_y,
                                     width=FOOD_SIZE, height=FOOD_SIZE, color=FOOD_COLOR)
        for segment in self.snake_body:
            arcade.draw_rectangle_filled(center_x=segment['x'], center_y=segment['y'],
                                         width=SNAKE_SIZE, height=SNAKE_SIZE, color=SNAKE_COLOR)
        self.draw_score()

    def on_update(self, delta_time):
        if self.game_over or self.ai_controlled:
            return

        self.movement_timer += delta_time
        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move

            if self.snake_dx_step == 0 and self.snake_dy_step == 0:
                return

            head = self.snake_body[0]
            new_head_x = head['x'] + self.snake_dx_step
            new_head_y = head['y'] + self.snake_dy_step

            # Colisiones para el jugador humano (usará las nuevas funciones específicas)
            # check_body_from_index=1 porque la cabeza aún está en su posición antigua en snake_body
            if self._is_wall_collision(new_head_x, new_head_y):
                self.game_over = True
                self.last_collision_type = "wall"
                return
            # El 1 es importante aquí
            if self._is_body_collision(new_head_x, new_head_y, 1):
                self.game_over = True
                self.last_collision_type = "self"
                return

            new_head = {'x': new_head_x, 'y': new_head_y}
            ate_food = False
            if abs(new_head_x - self.food_x) < (STEP_SIZE // 2) and \
               abs(new_head_y - self.food_y) < (STEP_SIZE // 2):
                ate_food = True
                self.score += POINTS_PER_FOOD

            self.snake_body.insert(0, new_head)
            if not ate_food:
                self.snake_body.pop()
            else:
                self._place_food()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.training_should_stop = True
            return

        if self.game_over:
            if key == arcade.key.SPACE and not self.ai_controlled:
                self.setup()
            return

        if self.ai_controlled:
            return

        if key == arcade.key.UP:
            if self.snake_dy_step != -STEP_SIZE:
                self.snake_dy_step = STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.DOWN:
            if self.snake_dy_step != STEP_SIZE:
                self.snake_dy_step = -STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.LEFT:
            if self.snake_dx_step != STEP_SIZE:
                self.snake_dx_step = -STEP_SIZE
                self.snake_dy_step = 0
        elif key == arcade.key.RIGHT:
            if self.snake_dx_step != -STEP_SIZE:
                self.snake_dx_step = STEP_SIZE
                self.snake_dy_step = 0

    def on_close(self):
        self.training_should_stop = True


def main_human_game():
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT,
                     "Snake v9 - Humano", ai_controlled=False)
    arcade.run()


if __name__ == "__main__":
    main_human_game()
