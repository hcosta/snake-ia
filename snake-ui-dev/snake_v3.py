# snake_v3_corregido.py

import arcade

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v3 (Corregido)"

# --- Constantes de la Serpiente ---
SNAKE_SIZE = 20
# SNAKE_SPEED ya no define la velocidad en píxeles por frame directamente,
# sino el tamaño del "paso" que da la serpiente en cada movimiento.
# La velocidad real dependerá de TIME_PER_MOVE.
STEP_SIZE = SNAKE_SIZE # Renombramos SNAKE_SPEED a STEP_SIZE para claridad, representa cuánto avanza por movimiento

class SnakeGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        # --- Atributos de la Serpiente ---
        self.snake_x = SCREEN_WIDTH // 2
        self.snake_y = SCREEN_HEIGHT // 2
        self.snake_dx_step = 0 # Cambio en X por movimiento (en unidades de STEP_SIZE)
        self.snake_dy_step = 0 # Cambio en Y por movimiento (en unidades de STEP_SIZE)
        
        # --- Control del Tiempo de Movimiento ---
        self.movement_timer = 0.0
        # Ajusta este valor para cambiar la velocidad de la serpiente:
        # Un valor más alto significa movimientos más lentos (más tiempo entre movimientos).
        # Un valor más bajo significa movimientos más rápidos.
        # Por ejemplo, 0.1 significa 10 movimientos por segundo.
        self.time_per_move = 0.15 # segundos

    def on_draw(self):
        self.clear()
        arcade.draw_rectangle_filled(
            center_x=self.snake_x,
            center_y=self.snake_y,
            width=SNAKE_SIZE,
            height=SNAKE_SIZE,
            color=arcade.color.RED_DEVIL
        )

    def on_update(self, delta_time):
        # Acumulamos el tiempo transcurrido desde el último frame
        self.movement_timer += delta_time

        # Si ha pasado suficiente tiempo para el siguiente movimiento
        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move # Reseteamos el temporizador para el próximo intervalo

            # Actualizamos la posición de la cabeza de la serpiente
            # La serpiente se mueve una cantidad de STEP_SIZE en la dirección dada
            self.snake_x += self.snake_dx_step
            self.snake_y += self.snake_dy_step

            # Aquí añadiremos lógica de colisión con paredes y consigo misma en el futuro

    def on_key_press(self, key, modifiers):
        """
        Este método se llama cada vez que el usuario presiona una tecla.
        'key' es la tecla que se presionó.
        'modifiers' indica si se presionaron teclas modificadoras como Shift, Ctrl, Alt.
        """

        # Cambiamos la dirección de la serpiente basándonos en la tecla presionada.
        # self.snake_dx_step y self.snake_dy_step ahora almacenan la dirección del próximo movimiento.
        # STEP_SIZE se usa para determinar la magnitud del movimiento cuando este ocurre.

        if key == arcade.key.UP:
            # Solo nos movemos hacia arriba si no nos estábamos moviendo hacia abajo
            # Comparamos la dirección actual de dy_step. Si es STEP_SIZE (moviéndose abajo), no permitir UP.
            # Nota: snake_dy_step será -STEP_SIZE para ABAJO y STEP_SIZE para ARRIBA
            if self.snake_dy_step != STEP_SIZE: 
                self.snake_dy_step = STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.DOWN:
            # Solo nos movemos hacia abajo si no nos estábamos moviendo hacia arriba
            if self.snake_dy_step != -STEP_SIZE:
                self.snake_dy_step = -STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.LEFT:
            # Solo nos movemos hacia la izquierda si no nos estábamos moviendo hacia la derecha
            if self.snake_dx_step != STEP_SIZE:
                self.snake_dx_step = -STEP_SIZE
                self.snake_dy_step = 0
        elif key == arcade.key.RIGHT:
            # Solo nos movemos hacia la derecha si no nos estábamos moviendo hacia la izquierda
            if self.snake_dx_step != -STEP_SIZE:
                self.snake_dx_step = STEP_SIZE
                self.snake_dy_step = 0

def main():
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()