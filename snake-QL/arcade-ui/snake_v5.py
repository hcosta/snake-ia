# snake_v5_corregido.py

import arcade
import random

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v5 (Lógica v4 integrada)"

# --- Constantes del Juego ---
SNAKE_SIZE = 20
STEP_SIZE = SNAKE_SIZE # Magnitud del movimiento en cada paso (alineado con el tamaño)
FOOD_SIZE = SNAKE_SIZE 

# --- Colores ---
BACKGROUND_COLOR = arcade.color.AMAZON
SNAKE_COLOR = arcade.color.RED_DEVIL
FOOD_COLOR = arcade.color.APPLE_GREEN

class SnakeGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(BACKGROUND_COLOR)

        # --- Atributos para el control de tiempo (de v4 corregida) ---
        self.movement_timer = 0.0
        # Ajusta este valor para cambiar la velocidad: más alto = más lento.
        self.time_per_move = 0.1 # Ejemplo: 10 movimientos por segundo

        # Los atributos de la serpiente y comida se inicializan en setup()
        self.snake_body = []
        self.snake_dx_step = 0
        self.snake_dy_step = 0
        self.food_x = 0
        self.food_y = 0
        
        self.setup() # Método para inicializar o reiniciar el juego

    def setup(self):
        """ Configura el juego para un nuevo inicio o reinicio. """
        # Calcular la posición inicial de la cabeza alineada a la cuadrícula
        head_start_x = (SCREEN_WIDTH // 2 // STEP_SIZE) * STEP_SIZE + (STEP_SIZE // 2)
        head_start_y = (SCREEN_HEIGHT // 2 // STEP_SIZE) * STEP_SIZE + (STEP_SIZE // 2)

        # Inicializar la serpiente con una cabeza y dos segmentos de cuerpo, alineados
        self.snake_body = [
            {'x': head_start_x, 'y': head_start_y},                               # Cabeza
            {'x': head_start_x - STEP_SIZE, 'y': head_start_y},                   # Segmento 1
            {'x': head_start_x - 2 * STEP_SIZE, 'y': head_start_y}                # Segmento 2
        ]
        self.snake_dx_step = 0 # Dirección inicial en X (quieta)
        self.snake_dy_step = 0 # Dirección inicial en Y (quieta)
        
        self._place_food()

    def _place_food(self):
        """ Coloca la comida en una posición aleatoria alineada a la cuadrícula,
            asegurándose de que no aparezca sobre la serpiente. """
        placed_on_snake = True
        while placed_on_snake:
            # Elige una casilla aleatoria con margen
            max_grid_cols = (SCREEN_WIDTH // SNAKE_SIZE) - 2 # Número de columnas disponibles con margen de 1
            max_grid_rows = (SCREEN_HEIGHT // SNAKE_SIZE) - 2 # Número de filas disponibles con margen de 1
            
            random_grid_col = random.randint(1, max_grid_cols) # Índice de columna (1 a max_grid_cols)
            random_grid_row = random.randint(1, max_grid_rows) # Índice de fila (1 a max_grid_rows)

            # Calcula el centro de la casilla seleccionada
            self.food_x = (random_grid_col * SNAKE_SIZE) + (SNAKE_SIZE // 2)
            self.food_y = (random_grid_row * SNAKE_SIZE) + (SNAKE_SIZE // 2)

            placed_on_snake = False
            for segment in self.snake_body:
                if segment['x'] == self.food_x and segment['y'] == self.food_y:
                    placed_on_snake = True
                    break # Vuelve a intentar colocar la comida

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

        # Dibujar cada segmento de la serpiente
        for segment in self.snake_body:
            arcade.draw_rectangle_filled(
                center_x=segment['x'],
                center_y=segment['y'],
                width=SNAKE_SIZE,
                height=SNAKE_SIZE,
                color=SNAKE_COLOR
            )

    def on_update(self, delta_time):
        self.movement_timer += delta_time

        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move # Resetea el temporizador para el próximo movimiento

            # Solo mover la serpiente si tiene una dirección (no está quieta)
            if self.snake_dx_step == 0 and self.snake_dy_step == 0:
                return

            # Calcular la nueva posición de la cabeza
            new_head_x = self.snake_body[0]['x'] + self.snake_dx_step
            new_head_y = self.snake_body[0]['y'] + self.snake_dy_step

            ate_food = False
            # Comprobar colisión con la comida (la nueva cabeza alcanza la comida)
            # Comparamos si los centros de la nueva cabeza y la comida están en la misma casilla.
            # Usamos una pequeña tolerancia por si acaso, aunque con STEP_SIZE deberían ser exactos.
            if abs(new_head_x - self.food_x) < (STEP_SIZE / 2) and \
               abs(new_head_y - self.food_y) < (STEP_SIZE / 2):
                ate_food = True
            
            # Crear la nueva cabeza
            new_head = {'x': new_head_x, 'y': new_head_y}
            
            # Añadir la nueva cabeza al principio del cuerpo de la serpiente
            self.snake_body.insert(0, new_head)

            # Si no comió, el último segmento desaparece (la serpiente se mueve)
            if not ate_food:
                self.snake_body.pop()
            else:
                # Si comió, no quitamos la cola (la serpiente crece) y colocamos nueva comida
                self._place_food()

    def on_key_press(self, key, modifiers):
        # Cambiar la dirección basada en la tecla presionada, usando STEP_SIZE
        if key == arcade.key.UP:
            if self.snake_dy_step != -STEP_SIZE: # Evitar moverse hacia abajo si ya va hacia arriba
                self.snake_dy_step = STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.DOWN:
            if self.snake_dy_step != STEP_SIZE: # Evitar moverse hacia arriba si ya va hacia abajo
                self.snake_dy_step = -STEP_SIZE
                self.snake_dx_step = 0
        elif key == arcade.key.LEFT:
            if self.snake_dx_step != STEP_SIZE: # Evitar moverse a la derecha si ya va a la izquierda
                self.snake_dx_step = -STEP_SIZE
                self.snake_dy_step = 0
        elif key == arcade.key.RIGHT:
            if self.snake_dx_step != -STEP_SIZE: # Evitar moverse a la izquierda si ya va a la derecha
                self.snake_dx_step = STEP_SIZE
                self.snake_dy_step = 0

def main():
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()