# snake_v7_corregido.py

import arcade
import random

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v7 (Lógica v6 integrada)"

# --- Constantes del Juego ---
SNAKE_SIZE = 20
STEP_SIZE = SNAKE_SIZE  # Magnitud del movimiento en cada paso
FOOD_SIZE = SNAKE_SIZE
POINTS_PER_FOOD = 10  # Puntos que se ganan por cada comida

# --- Colores ---
BACKGROUND_COLOR = arcade.color.AMAZON
SNAKE_COLOR = arcade.color.RED_DEVIL
FOOD_COLOR = arcade.color.APPLE_GREEN
GAMEOVER_COLOR = arcade.color.BLACK_LEATHER_JACKET # Color de Game Over (ligeramente modificado)
SCORE_COLOR = arcade.color.BLACK

class SnakeGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(BACKGROUND_COLOR)

        # Atributos para el control de tiempo (de v6 corregida)
        self.movement_timer = 0.0
        self.time_per_move = 0.1  # Movimientos por segundo (ej: 0.1 para 10 mov/seg)

        self.game_over = False
        self.score = 0
        # Los demás atributos se inicializan en setup()
        self.snake_body = []
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self.food_x = 0
        self.food_y = 0

        self.setup()

    def setup(self):
        """ Configura el juego para un nuevo inicio o reinicio. """
        self.game_over = False
        self.score = 0  # Reiniciamos la puntuación

        # Calcular la posición inicial de la cabeza alineada a la cuadrícula
        head_start_x = (SCREEN_WIDTH // 2 // STEP_SIZE) * STEP_SIZE + (STEP_SIZE // 2)
        head_start_y = (SCREEN_HEIGHT // 2 // STEP_SIZE) * STEP_SIZE + (STEP_SIZE // 2)

        self.snake_body = [
            {'x': head_start_x, 'y': head_start_y},
            {'x': head_start_x - STEP_SIZE, 'y': head_start_y},
            {'x': head_start_x - 2 * STEP_SIZE, 'y': head_start_y}
        ]
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self._place_food()

    def _place_food(self):
        """ Coloca la comida en una posición aleatoria alineada a la cuadrícula.
            (Método de snake_v7.py usando randrange, que es correcto)
        """
        placed_on_snake = True
        while placed_on_snake:
            # random.randrange(start, stop, step) -> stop es exclusivo
            # Esto genera centros de casillas desde la primera hasta la penúltima.
            self.food_x = random.randrange(STEP_SIZE // 2, SCREEN_WIDTH - STEP_SIZE // 2, STEP_SIZE)
            self.food_y = random.randrange(STEP_SIZE // 2, SCREEN_HEIGHT - STEP_SIZE // 2, STEP_SIZE)

            placed_on_snake = False
            for segment in self.snake_body:
                if abs(segment['x'] - self.food_x) < STEP_SIZE // 2 and \
                   abs(segment['y'] - self.food_y) < STEP_SIZE // 2:
                    placed_on_snake = True
                    break

    def draw_game_over(self):
        """ Dibuja el mensaje de Game Over y la puntuación final. """
        arcade.draw_text(
            "GAME OVER",
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2,
            GAMEOVER_COLOR,
            font_size=50,
            anchor_x="center",
            anchor_y="center"
        )
        arcade.draw_text(
            f"Puntuación Final: {self.score}", # Mostrar puntuación final
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2 - 50,
            GAMEOVER_COLOR,
            font_size=20,
            anchor_x="center",
            anchor_y="center"
        )
        arcade.draw_text(
            "Pulsa ESPACIO para reiniciar",
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2 - 100, # Ajustar posición
            GAMEOVER_COLOR,
            font_size=20,
            anchor_x="center",
            anchor_y="center"
        )

    def draw_score(self):
        """ Dibuja la puntuación actual en la pantalla. """
        score_text = f"Puntuación: {self.score}"
        arcade.draw_text(
            score_text,
            10,  # Posición X
            SCREEN_HEIGHT - 30,  # Posición Y
            SCORE_COLOR,
            font_size=20
        )

    def on_draw(self):
        self.clear()

        if self.game_over:
            self.draw_game_over()
            return

        self.draw_score()  # Dibujar puntuación

        # Dibujar comida
        arcade.draw_rectangle_filled(
            center_x=self.food_x, center_y=self.food_y,
            width=FOOD_SIZE, height=FOOD_SIZE, color=FOOD_COLOR
        )

        # Dibujar serpiente
        for segment in self.snake_body:
            arcade.draw_rectangle_filled(
                center_x=segment['x'], center_y=segment['y'],
                width=SNAKE_SIZE, height=SNAKE_SIZE, color=SNAKE_COLOR
            )

    def on_update(self, delta_time):
        if self.game_over:
            return

        self.movement_timer += delta_time
        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move

            if self.snake_dx_step == 0 and self.snake_dy_step == 0:
                return # No hay movimiento si no hay dirección

            head = self.snake_body[0]
            new_head_x = head['x'] + self.snake_dx_step
            new_head_y = head['y'] + self.snake_dy_step

            # 1. Comprobar colisión con paredes
            if not (STEP_SIZE // 2 <= new_head_x <= SCREEN_WIDTH - STEP_SIZE // 2 and
                    STEP_SIZE // 2 <= new_head_y <= SCREEN_HEIGHT - STEP_SIZE // 2):
                self.game_over = True
                return

            # 2. Comprobar colisión consigo misma
            for i in range(1, len(self.snake_body)):
                segment = self.snake_body[i]
                if abs(new_head_x - segment['x']) < STEP_SIZE // 2 and \
                   abs(new_head_y - segment['y']) < STEP_SIZE // 2:
                    self.game_over = True
                    return

            ate_food = False
            # Comprobar colisión con la comida
            if abs(new_head_x - self.food_x) < (STEP_SIZE // 2) and \
               abs(new_head_y - self.food_y) < (STEP_SIZE // 2):
                ate_food = True
                self.score += POINTS_PER_FOOD  # Incrementar puntuación

            # Crear la nueva cabeza
            new_head = {'x': new_head_x, 'y': new_head_y}
            # Añadir la nueva cabeza al principio
            self.snake_body.insert(0, new_head)

            if not ate_food:
                self.snake_body.pop()  # Quitar la cola si no comió
            else:
                self._place_food()  # Si comió, colocar nueva comida (y no quitar cola)

    def on_key_press(self, key, modifiers):
        if self.game_over:
            if key == arcade.key.SPACE:
                self.setup() # Reiniciar el juego
            return

        # Cambiar dirección usando STEP_SIZE
        if key == arcade.key.UP:
            if self.snake_dy_step != -STEP_SIZE: # No estaba yendo hacia ABAJO
                self.snake_dy_step = STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.DOWN:
            if self.snake_dy_step != STEP_SIZE: # No estaba yendo hacia ARRIBA
                self.snake_dy_step = -STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.LEFT:
            if self.snake_dx_step != STEP_SIZE: # No estaba yendo DERECHA
                self.snake_dx_step = -STEP_SIZE
                self.snake_dy_step = 0
        elif key == arcade.key.RIGHT:
            if self.snake_dx_step != -STEP_SIZE: # No estaba yendo IZQUIERDA
                self.snake_dx_step = STEP_SIZE
                self.snake_dy_step = 0

def main():
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()