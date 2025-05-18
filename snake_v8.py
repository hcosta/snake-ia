# snake_v8_ia_ready.py
import arcade
import random
import numpy as np

# --- Constantes de la Pantalla ---
# Asumiré que vuelves a 300x300 para estas pruebas, pero ajústalo si es necesario
SCREEN_WIDTH = 250
SCREEN_HEIGHT = 250
SCREEN_TITLE = "Snake IA"

# --- Constantes del Juego ---
SNAKE_SIZE = 20
STEP_SIZE = SNAKE_SIZE  # Asegúrate que STEP_SIZE es igual a SNAKE_SIZE
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

# Recompensas (ajusta según tus últimas pruebas exitosas)
# Estos son ejemplos, usa los que te estaban dando buenos resultados
# con EPSILON_MIN muy bajo.
REWARD_FOOD = 30
REWARD_GAME_OVER = -100
REWARD_MOVE = -0.001
REWARD_STUCK_LOOP = -150


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
        self.snake_dx_step = 0  # Dirección actual de movimiento en X
        self.snake_dy_step = 0  # Dirección actual de movimiento en Y
        self.food_x = 0
        self.food_y = 0

        self.training_should_stop = False
        self.setup()

    def setup(self):
        self.game_over = False
        self.score = 0
        # Posición inicial de la cabeza (centrada)
        head_start_x = (SCREEN_WIDTH // 2 // STEP_SIZE) * \
            STEP_SIZE + (STEP_SIZE // 2)
        head_start_y = (SCREEN_HEIGHT // 2 // STEP_SIZE) * \
            STEP_SIZE + (STEP_SIZE // 2)
        self.snake_body = [
            {'x': head_start_x, 'y': head_start_y},
            {'x': head_start_x - STEP_SIZE, 'y': head_start_y},
            {'x': head_start_x - 2 * STEP_SIZE, 'y': head_start_y}
        ]
        # Al inicio, la serpiente está quieta hasta que se toma una acción o tecla.
        # Para la IA, la primera acción establecerá la dirección.
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self._place_food()
        self.training_should_stop = False
        return self.get_state()  # Devuelve el estado inicial

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

    def _is_general_collision(self, x, y, check_body_from_index=0):
        # Colisión con bordes
        x_coord_min = STEP_SIZE // 2
        y_coord_min = STEP_SIZE // 2
        x_coord_max_center = SCREEN_WIDTH - (STEP_SIZE // 2)
        y_coord_max_center = SCREEN_HEIGHT - (STEP_SIZE // 2)

        if not (x_coord_min <= x <= x_coord_max_center and
                y_coord_min <= y <= y_coord_max_center):
            return True

        # Colisión con el cuerpo
        # `check_body_from_index` se usa para evitar chequear la cabeza contra sí misma
        # si la cabeza ya se ha movido a la nueva posición antes de llamar a esta función.
        # Si se llama para una posición futura, `check_body_from_index=0` o `1` es apropiado
        # si el cuerpo aún no se ha actualizado.
        for i in range(check_body_from_index, len(self.snake_body)):
            segment = self.snake_body[i]
            if abs(x - segment['x']) < STEP_SIZE // 2 and \
               abs(y - segment['y']) < STEP_SIZE // 2:
                return True
        return False

    # NUEVA FUNCIÓN AUXILIAR para detectar callejones sin salida
    def _is_immediate_trap(self, x_start, y_start, dx_forward, dy_forward, check_body_from_index_for_trap_check=0):
        """
        Verifica si una celda (x_start, y_start) es una trampa de 1 paso.
        Una trampa significa que la celda está libre, pero todas sus salidas
        (recto, izquierda relativa, derecha relativa) desde esa celda están bloqueadas.
        dx_forward, dy_forward indican la dirección "hacia adelante" desde la perspectiva
        de la serpiente al entrar en (x_start, y_start).
        check_body_from_index_for_trap_check se pasa a _is_general_collision.
        Generalmente será 0 si evaluamos una posición futura donde la cabeza aún no se ha añadido al cuerpo.
        """
        # 1. La celda (x_start, y_start) en sí misma no debe ser una colisión para ser una "entrada" a una trampa
        #    Esta comprobación se hace antes de llamar a _is_immediate_trap usualmente.

        # 2. Comprobar las 3 salidas desde (x_start, y_start)
        #    La dirección "hacia atrás" no se considera una salida válida para escapar de una trampa.

        # Salida Recta desde (x_start, y_start)
        next_x_straight = x_start + dx_forward
        next_y_straight = y_start + dy_forward
        straight_blocked = self._is_general_collision(
            next_x_straight, next_y_straight, check_body_from_index_for_trap_check)

        # Salida Izquierda Relativa desde (x_start, y_start)
        # Rotación antihoraria: dx_new = -dy_old, dy_new = dx_old
        dx_left_rel = -dy_forward
        dy_left_rel = dx_forward
        next_x_left_rel = x_start + dx_left_rel
        next_y_left_rel = y_start + dy_left_rel
        left_rel_blocked = self._is_general_collision(
            next_x_left_rel, next_y_left_rel, check_body_from_index_for_trap_check)

        # Salida Derecha Relativa desde (x_start, y_start)
        # Rotación horaria: dx_new = dy_old, dy_new = -dx_old
        dx_right_rel = dy_forward
        dy_right_rel = -dx_forward
        next_x_right_rel = x_start + dx_right_rel
        next_y_right_rel = y_start + dy_right_rel
        right_rel_blocked = self._is_general_collision(
            next_x_right_rel, next_y_right_rel, check_body_from_index_for_trap_check)

        return straight_blocked and left_rel_blocked and right_rel_blocked

    def get_state(self):
        head = self.snake_body[0]

        # current_dx_eff y current_dy_eff representan la dirección de movimiento "efectiva" o "intencionada"
        # Si la serpiente está quieta (inicio del juego), asumimos que "intentará" ir a la derecha
        # para tener una referencia para las direcciones relativas.
        current_dx_eff, current_dy_eff = self.snake_dx_step, self.snake_dy_step
        if current_dx_eff == 0 and current_dy_eff == 0:  # Inicio del juego, sin movimiento previo
            current_dx_eff = STEP_SIZE  # Asumir movimiento a la derecha como base
            current_dy_eff = 0

        # --- Peligro Inmediato (1 paso) ---
        # Coordenadas para el chequeo de peligro recto
        check_x_straight = head['x'] + current_dx_eff
        check_y_straight = head['y'] + current_dy_eff
        # check_body_from_index=1 para no chocar con la cabeza (asumiendo que aún no se ha movido)
        peligro_recto = self._is_general_collision(
            check_x_straight, check_y_straight, 1)

        # Coordenadas para el chequeo de peligro a la izquierda relativa
        # Izquierda relativa: rotar vector de dirección (-dy, dx)
        dx_left = -current_dy_eff
        dy_left = current_dx_eff
        check_x_left = head['x'] + dx_left
        check_y_left = head['y'] + dy_left
        peligro_izquierda_rel = self._is_general_collision(
            check_x_left, check_y_left, 1)

        # Coordenadas para el chequeo de peligro a la derecha relativa
        # Derecha relativa: rotar vector de dirección (dy, -dx)
        dx_right = current_dy_eff
        dy_right = -current_dx_eff
        check_x_right = head['x'] + dx_right
        check_y_right = head['y'] + dy_right
        peligro_derecha_rel = self._is_general_collision(
            check_x_right, check_y_right, 1)

        # --- NUEVO: Detección de Callejón Sin Salida (Trampa de 1 paso) ---
        # Si moverse en una dirección es seguro (no hay peligro inmediato),
        # ¿esa casilla segura es a su vez una trampa (sin salidas)?
        # Usamos check_body_from_index=0 para _is_immediate_trap porque evaluamos
        # una casilla futura donde la cabeza no se ha movido aún, y la serpiente
        # podría chocar con su propio cuerpo al intentar salir de esa trampa.

        callejon_recto = 0
        if not peligro_recto:  # Si moverse recto es seguro
            # (check_x_straight, check_y_straight) es la casilla a 1 paso recto
            # current_dx_eff, current_dy_eff es la dirección para llegar ahí (y para salir de ahí recto)
            callejon_recto = self._is_immediate_trap(
                check_x_straight, check_y_straight, current_dx_eff, current_dy_eff, 0)

        callejon_izquierda_rel = 0
        if not peligro_izquierda_rel:  # Si moverse a la izquierda relativa es seguro
            # (check_x_left, check_y_left) es la casilla a 1 paso a la izquierda relativa
            # (dx_left, dy_left) es la dirección para llegar ahí (y para salir de ahí recto)
            callejon_izquierda_rel = self._is_immediate_trap(
                check_x_left, check_y_left, dx_left, dy_left, 0)

        callejon_derecha_rel = 0
        if not peligro_derecha_rel:  # Si moverse a la derecha relativa es seguro
            # (check_x_right, check_y_right) es la casilla a 1 paso a la derecha relativa
            # (dx_right, dy_right) es la dirección para llegar ahí (y para salir de ahí recto)
            callejon_derecha_rel = self._is_immediate_trap(
                check_x_right, check_y_right, dx_right, dy_right, 0)

        # --- Dirección de la Comida ---
        food_dir_x = np.sign(self.food_x - head['x'])
        food_dir_y = np.sign(self.food_y - head['y'])

        # --- Dirección Actual de la Serpiente (Categorizada) ---
        # Usa la dirección real de la serpiente (self.snake_dx_step, self.snake_dy_step)
        # no la efectiva, ya que esta es una característica de la serpiente, no una evaluación de una acción futura.
        dir_category = 4  # Categoría para "sin movimiento" o estado inicial
        if self.snake_dx_step == STEP_SIZE:
            dir_category = 3  # Derecha
        elif self.snake_dx_step == -STEP_SIZE:
            dir_category = 2  # Izquierda
        elif self.snake_dy_step == STEP_SIZE:
            dir_category = 0  # Arriba
        elif self.snake_dy_step == -STEP_SIZE:
            dir_category = 1  # Abajo

        state = (
            # Peligros inmediatos
            int(peligro_recto),
            int(peligro_izquierda_rel),
            int(peligro_derecha_rel),
            # NUEVOS ESTADOS: Callejones sin salida
            int(callejon_recto),
            int(callejon_izquierda_rel),
            int(callejon_derecha_rel),
            # Dirección de la comida
            int(food_dir_x),
            int(food_dir_y),
            # Dirección actual de la serpiente
            dir_category
        )
        return state

    def step(self, action):  # El parámetro action es el índice de la acción (0,1,2,3)
        reward = REWARD_MOVE
        done = False
        info = {}  # Para información de depuración adicional, si fuera necesario

        # Obtener el cambio de coordenadas (dx, dy) para la acción seleccionada
        potential_move = ACTION_MAP.get(action)

        requested_dx = self.snake_dx_step  # Mantener dirección actual por defecto
        requested_dy = self.snake_dy_step

        if potential_move:
            # Evitar que la serpiente se mueva en la dirección opuesta a su movimiento actual
            # Esto solo aplica si la serpiente ya está en movimiento.
            is_opposite_move = False
            if self.snake_dx_step != 0 or self.snake_dy_step != 0:  # Si la serpiente se está moviendo
                if potential_move['dx'] == -self.snake_dx_step and self.snake_dx_step != 0:
                    is_opposite_move = True
                if potential_move['dy'] == -self.snake_dy_step and self.snake_dy_step != 0:
                    is_opposite_move = True

            if not is_opposite_move:
                requested_dx = potential_move['dx']
                requested_dy = potential_move['dy']

        # Actualizar la dirección de la serpiente
        self.snake_dx_step = requested_dx
        self.snake_dy_step = requested_dy

        # Si después de procesar la acción, la serpiente no tiene una dirección de movimiento
        # (esto podría pasar si la acción era opuesta y no había movimiento previo,
        # o si la acción no es válida, aunque ACTION_MAP siempre debería dar una),
        # no hacemos nada más en este paso. Esto es importante para el inicio.
        if self.snake_dx_step == 0 and self.snake_dy_step == 0:
            # Si no hay movimiento, simplemente devolvemos el estado actual sin cambios mayores.
            # Esto ocurre al inicio si la primera acción no inicia el movimiento.
            current_observation = self.get_state()
            # No hay movimiento, así que solo se aplica la recompensa por movimiento (si es negativa).
            # No hay colisión ni comida.
            return current_observation, reward, done, info

        # Calcular la nueva posición de la cabeza
        head = self.snake_body[0]
        new_head_x = head['x'] + self.snake_dx_step
        new_head_y = head['y'] + self.snake_dy_step

        # Comprobar colisiones (bordes o cuerpo)
        # Aquí usamos check_body_from_index=0 porque la cabeza aún no se ha "movido"
        # a la nueva lista snake_body. Estamos evaluando la colisión de la *nueva* cabeza.
        if self._is_general_collision(new_head_x, new_head_y, 0):
            done = True
            reward += REWARD_GAME_OVER
            self.game_over = True

        if self.game_over:  # Si hay colisión, terminar el episodio
            # Devolver el estado actual (que será el estado justo antes del game over)
            # y la recompensa final.
            return self.get_state(), reward, done, info

        # Si no hay colisión, mover la serpiente
        new_head_segment = {'x': new_head_x, 'y': new_head_y}

        # Comprobar si se comió la comida
        ate_food = False
        if abs(new_head_x - self.food_x) < (STEP_SIZE // 2) and \
           abs(new_head_y - self.food_y) < (STEP_SIZE // 2):
            ate_food = True
            self.score += POINTS_PER_FOOD
            reward += REWARD_FOOD

        # Insertar la nueva cabeza
        self.snake_body.insert(0, new_head_segment)

        if not ate_food:
            self.snake_body.pop()  # Quitar el último segmento si no se comió
        else:
            self._place_food()  # Colocar nueva comida
            # Comprobar si el juego se ha "ganado" (pantalla llena)
            if len(self.snake_body) >= (SCREEN_WIDTH / STEP_SIZE) * (SCREEN_HEIGHT / STEP_SIZE):
                done = True  # Juego ganado
                self.game_over = True  # Considerar como game over para terminar episodio

        # Obtener el nuevo estado después del movimiento
        next_observation = self.get_state()
        return next_observation, reward, done, info

    def draw_game_over(self):
        arcade.draw_text("GAME OVER", SCREEN_WIDTH / 2, (SCREEN_HEIGHT / 2) + 50, GAMEOVER_COLOR,
                         font_size=20, anchor_x="center", anchor_y="center")
        arcade.draw_text(f"{self.score//10} puntos", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2,
                         GAMEOVER_COLOR, font_size=20, anchor_x="center", anchor_y="center")
        if not self.ai_controlled:
            arcade.draw_text("RESET = SPACE", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50,
                             GAMEOVER_COLOR, font_size=20, anchor_x="center", anchor_y="center")

    def draw_score(self):
        score_text = f"{self.score//10}"
        arcade.draw_text(score_text, 10, SCREEN_HEIGHT -
                         30, SCORE_COLOR, font_size=20)

    def on_draw(self):
        self.clear()
        if self.game_over:
            self.draw_game_over()
            return
        arcade.draw_rectangle_filled(center_x=self.food_x, center_y=self.food_y,
                                     width=FOOD_SIZE, height=FOOD_SIZE, color=FOOD_COLOR)
        for segment in self.snake_body:
            arcade.draw_rectangle_filled(center_x=segment['x'], center_y=segment['y'],
                                         width=SNAKE_SIZE, height=SNAKE_SIZE, color=SNAKE_COLOR)

        self.draw_score()

    def on_update(self, delta_time):
        # Esta función es principalmente para el control humano y el temporizador de movimiento.
        # Para la IA, la lógica de movimiento está en step().
        if self.game_over or self.ai_controlled:  # Si es IA o juego terminado, no hacer nada aquí
            return

        # Lógica de movimiento para el jugador humano (basada en tiempo)
        self.movement_timer += delta_time
        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move

            if self.snake_dx_step == 0 and self.snake_dy_step == 0:  # No moverse si no hay dirección
                return

            head = self.snake_body[0]
            new_head_x = head['x'] + self.snake_dx_step
            new_head_y = head['y'] + self.snake_dy_step

            # Colisiones para el jugador humano
            # Empezar chequeo desde el segundo segmento
            if self._is_general_collision(new_head_x, new_head_y, 1):
                self.game_over = True
                return

            # Mover serpiente para jugador humano
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
            print("Tecla ESCAPE detectada. Solicitando parada...")
            self.training_should_stop = True  # Bandera para el bucle de entrenamiento externo
            # No cerramos la ventana aquí, lo hará el trainer si es necesario.
            return

        if self.game_over:
            if key == arcade.key.SPACE and not self.ai_controlled:  # Reiniciar juego humano
                self.setup()
            return

        if self.ai_controlled:  # La IA no usa teclado para moverse durante el step()
            return

        # Control de teclado para el jugador humano
        if key == arcade.key.UP:
            if self.snake_dy_step != -STEP_SIZE:  # Evitar moverse en dirección opuesta
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
        print("Botón 'X' de la ventana detectado. Solicitando parada del proceso...")
        self.training_should_stop = True


def main_human_game():
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT,
                     "Snake - Humano", ai_controlled=False)
    arcade.run()


if __name__ == "__main__":
    main_human_game()
