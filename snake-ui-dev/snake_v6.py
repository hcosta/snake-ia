# snake_v6_corregido.py

import arcade
import random

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v6 (Lógica v5 integrada)"

# --- Constantes del Juego ---
SNAKE_SIZE = 20
STEP_SIZE = SNAKE_SIZE # Magnitud del movimiento en cada paso
FOOD_SIZE = SNAKE_SIZE 

# --- Colores ---
BACKGROUND_COLOR = arcade.color.AMAZON
SNAKE_COLOR = arcade.color.RED_DEVIL
FOOD_COLOR = arcade.color.APPLE_GREEN
GAMEOVER_COLOR = arcade.color.BLACK_LEATHER_JACKET # Un negro un poco más suave

class SnakeGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(BACKGROUND_COLOR)
        
        # Atributos para el control de tiempo (de v5 corregida)
        self.movement_timer = 0.0
        self.time_per_move = 0.1 # Movimientos por segundo (ej: 0.1 para 10 mov/seg)

        self.game_over = False
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
        """ Coloca la comida en una posición aleatoria alineada a la cuadrícula,
            asegurándose de que no aparezca sobre la serpiente.
            (Método mejorado de snake_v6.py usando randrange)
        """
        placed_on_snake = True
        while placed_on_snake:
            # Genera coordenadas para el centro de una casilla aleatoria
            # random.randrange(start, stop, step) -> stop es exclusivo
            self.food_x = random.randrange(STEP_SIZE // 2, SCREEN_WIDTH - STEP_SIZE // 2 + 1, STEP_SIZE)
            self.food_y = random.randrange(STEP_SIZE // 2, SCREEN_HEIGHT - STEP_SIZE // 2 + 1, STEP_SIZE)
            
            placed_on_snake = False
            for segment in self.snake_body:
                # Comprueba si el centro de la comida coincide con el centro de algún segmento
                if abs(segment['x'] - self.food_x) < STEP_SIZE // 2 and \
                   abs(segment['y'] - self.food_y) < STEP_SIZE // 2:
                    placed_on_snake = True
                    break
    
    def draw_game_over(self):
        """ Dibuja el mensaje de Game Over. """
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
            "Pulsa ESPACIO para reiniciar",
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT / 2 - 50, # Un poco más abajo
            GAMEOVER_COLOR,
            font_size=20,
            anchor_x="center",
            anchor_y="center"
        )

    def on_draw(self):
        self.clear()
        
        if self.game_over:
            self.draw_game_over()
            return # No dibujamos nada más si es game over

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
            return # No actualizamos nada si el juego ha terminado

        self.movement_timer += delta_time
        if self.movement_timer >= self.time_per_move:
            self.movement_timer -= self.time_per_move

            # Si la serpiente no tiene dirección, no hacer nada más en este paso de movimiento
            if self.snake_dx_step == 0 and self.snake_dy_step == 0:
                return

            head = self.snake_body[0]
            new_head_x = head['x'] + self.snake_dx_step
            new_head_y = head['y'] + self.snake_dy_step

            # 1. Comprobar colisión con paredes
            # Los centros de los segmentos deben estar dentro de los límites válidos
            # Límite izquierdo/inferior: STEP_SIZE / 2
            # Límite derecho/superior: SCREEN_WIDTH/HEIGHT - STEP_SIZE / 2
            if not (STEP_SIZE // 2 <= new_head_x <= SCREEN_WIDTH - STEP_SIZE // 2 and
                    STEP_SIZE // 2 <= new_head_y <= SCREEN_HEIGHT - STEP_SIZE // 2):
                self.game_over = True
                return 

            # 2. Comprobar colisión consigo misma (con la nueva posición de la cabeza)
            # Empezamos desde el índice 1 porque el 0 es la cabeza actual (antes de moverla)
            # pero la colisión es con el cuerpo existente ANTES de añadir la nueva cabeza y quitar la cola
            # (si la serpiente es muy corta, no hay colisión posible con el cuerpo que se va a mover)
            # Un chequeo más robusto es contra self.snake_body[0:-1] si se permite morder la cola que justo se movió
            # O contra self.snake_body[1:] si la cabeza no puede ocupar la casilla de la segunda pieza inmediatamente.
            # La implementación actual (range(1, len(self.snake_body))) compara la nueva cabeza
            # con todos los segmentos del cuerpo actual *antes* de que la cola se mueva o crezca.
            for i in range(len(self.snake_body)): # Comprobamos contra todos los segmentos actuales
                                                 # porque si la nueva cabeza aterriza donde está una parte del cuerpo
                                                 # (excepto la cola que se va a mover si no come), es game over.
                                                 # Si la longitud es 1, este bucle no hace nada (pero es raro)
                segment = self.snake_body[i]
                # Si la nueva cabeza está en la misma posición que un segmento existente Y
                # ese segmento no es la cola que está a punto de desaparecer (si no come)
                # Esta lógica es más simple: si la nueva cabeza ocupa CUALQUIER casilla de un segmento actual, es colisión
                # (a menos que sea una serpiente muy pequeña donde la "cola" es la cabeza anterior)
                # El código original de v6 es bueno para esto:
            for i in range(1, len(self.snake_body)): # Chequear contra el cuerpo, excluyendo la cabeza actual
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
            
            # Crear la nueva cabeza
            new_head = {'x': new_head_x, 'y': new_head_y}
            # Añadir la nueva cabeza al principio
            self.snake_body.insert(0, new_head)

            if not ate_food:
                self.snake_body.pop() # Quitar la cola si no comió
            else:
                self._place_food() # Si comió, colocar nueva comida (y no quitar cola)


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