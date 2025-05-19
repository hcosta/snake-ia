# snake_v4_corregido.py

import arcade
import random # Necesitaremos la librería random para la posición de la comida

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v4 (Lógica v3 integrada)"

# --- Constantes de la Serpiente ---
SNAKE_SIZE = 20
STEP_SIZE = SNAKE_SIZE # Renombramos SNAKE_SPEED para mayor claridad: es el tamaño del paso en cada movimiento

# --- Constantes de la Comida ---
FOOD_SIZE = SNAKE_SIZE 
FOOD_COLOR = arcade.color.APPLE_GREEN

class SnakeGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        # --- Atributos de la Serpiente ---
        # Aseguramos que la serpiente inicie en el centro de una casilla de la cuadrícula
        self.snake_x = (SCREEN_WIDTH // 2 // STEP_SIZE) * STEP_SIZE + STEP_SIZE // 2
        self.snake_y = (SCREEN_HEIGHT // 2 // STEP_SIZE) * STEP_SIZE + STEP_SIZE // 2
        
        self.snake_dx_step = 0 # Cambio en X por movimiento (en unidades de STEP_SIZE)
        self.snake_dy_step = 0 # Cambio en Y por movimiento (en unidades de STEP_SIZE)
        
        # --- Control del Tiempo de Movimiento (de la v3 corregida) ---
        self.movement_timer = 0.0
        # Ajusta este valor para cambiar la velocidad: más alto = más lento.
        # 0.15 significa ~6.6 movimientos por segundo.
        self.time_per_move = 0.15 

        # --- Atributos de la Comida ---
        self.food_x = 0
        self.food_y = 0
        self._place_food() # Colocamos la comida por primera vez

    def _place_food(self):
        """
        Método para colocar la comida en una posición aleatoria ALINEADA A LA CUADRÍCULA.
        Nos aseguramos de que no aparezca demasiado cerca de los bordes.
        (Esta función es la original de snake_v4.py, ya que es correcta y está bien implementada)
        """
        # Para asegurar la alineación, las posiciones deben ser múltiplos de SNAKE_SIZE.
        # Calculamos cuántas "casillas" caben en la pantalla, menos un margen.
        max_x_grid_index = (SCREEN_WIDTH // SNAKE_SIZE) - 2 
        max_y_grid_index = (SCREEN_HEIGHT // SNAKE_SIZE) - 2

        # Elegimos una casilla aleatoria (índice basado en las casillas disponibles con margen)
        # random.randint(1, N) elige un número entre 1 y N inclusive.
        # Aquí, random_casilla_x será el índice (1-based) de la columna de la cuadrícula,
        # excluyendo la columna 0 y la última columna.
        random_grid_col = random.randint(1, max_x_grid_index)
        random_grid_row = random.randint(1, max_y_grid_index)

        # Convertimos el índice de la casilla a coordenadas en píxeles para el CENTRO de la comida
        self.food_x = random_grid_col * SNAKE_SIZE + SNAKE_SIZE // 2
        self.food_y = random_grid_row * SNAKE_SIZE + SNAKE_SIZE // 2

    def on_draw(self):
        self.clear()
        
        # Dibujar la comida
        arcade.draw_rectangle_filled(
            center_x=self.food_x,
            center_y=self.food_y,
            width=FOOD_SIZE,
            height=FOOD_SIZE,
            color=FOOD_COLOR
        )

        # Dibujar la cabeza de la serpiente
        arcade.draw_rectangle_filled(
            center_x=self.snake_x,
            center_y=self.snake_y,
            width=SNAKE_SIZE,
            height=SNAKE_SIZE,
            color=arcade.color.RED_DEVIL
        )

    def on_update(self, delta_time):
        # Acumulamos el tiempo transcurrido (lógica de v3 corregida)
        self.movement_timer += delta_time

        # Si ha pasado suficiente tiempo para el siguiente movimiento
        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move # Restamos/reseteamos el tiempo del movimiento actual

            # Actualizamos la posición de la cabeza de la serpiente
            self.snake_x += self.snake_dx_step
            self.snake_y += self.snake_dy_step

            # Lógica de colisión con la comida (la añadiremos en la próxima lección)
            # if arcade.check_for_collision(serpiente_sprite, comida_sprite): ...
            # O, si usamos coordenadas directamente:
            # distancia_x = abs(self.snake_x - self.food_x)
            # distancia_y = abs(self.snake_y - self.food_y)
            # # La colisión ocurre si los centros están a menos de SNAKE_SIZE de distancia
            # # (ya que food_size es igual a snake_size, y ambos son cuadrados)
            # if distancia_x < SNAKE_SIZE and distancia_y < SNAKE_SIZE:
            # # Una comprobación más precisa si las formas son exactas y no solo AABB:
            # # if self.snake_x == self.food_x and self.snake_y == self.food_y: (si ambos son centros de la misma casilla)
            #     print("¡Comida!")
            #     self._place_food() # Colocar nueva comida
            #     # Aquí también se debería incrementar el tamaño de la serpiente, etc.


    def on_key_press(self, key, modifiers):
        """
        Maneja los eventos de presión de teclas para cambiar la dirección de la serpiente.
        Usa STEP_SIZE para la magnitud del movimiento.
        """
        if key == arcade.key.UP:
            # Solo nos movemos hacia arriba si no nos estábamos moviendo hacia abajo
            if self.snake_dy_step != -STEP_SIZE: 
                self.snake_dy_step = STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.DOWN:
            # Solo nos movemos hacia abajo si no nos estábamos moviendo hacia arriba
            if self.snake_dy_step != STEP_SIZE:
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