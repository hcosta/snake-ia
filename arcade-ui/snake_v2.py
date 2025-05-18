# snake_v2.py

import arcade

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v2"

# --- Constantes de la Serpiente ---
SNAKE_SIZE = 20
SNAKE_SPEED = 10 # Ajustaremos esta velocidad más adelante

class SnakeGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        arcade.set_background_color(arcade.color.AMAZON)

        # --- Atributos de la Serpiente ---
        # Posición inicial de la cabeza de la serpiente (centro de la pantalla)
        self.snake_x = SCREEN_WIDTH // 2
        self.snake_y = SCREEN_HEIGHT // 2

        # Dirección inicial del movimiento de la serpiente
        # dx (delta x) y dy (delta y) indican cuánto cambia la posición en cada eje por actualización.
        self.snake_dx = 0  # No se mueve horizontalmente al inicio
        self.snake_dy = 0  # No se mueve verticalmente al inicio
        
        # Para que la serpiente empiece moviéndose, vamos a darle una dirección inicial.
        # Por ejemplo, que empiece moviéndose hacia la derecha:
        self.snake_dx = SNAKE_SPEED

    def on_draw(self):
        self.clear()

        # Dibujamos la cabeza de la serpiente
        # arcade.draw_rectangle_filled dibuja un rectángulo relleno.
        # Parámetros: centro_x, centro_y, ancho, alto, color
        arcade.draw_rectangle_filled(
            center_x=self.snake_x,
            center_y=self.snake_y,
            width=SNAKE_SIZE,
            height=SNAKE_SIZE,
            color=arcade.color.RED_DEVIL # Un color llamativo para la cabeza
        )

    def on_update(self, delta_time):
        # Actualizamos la posición de la cabeza de la serpiente
        # Sumamos el cambio en x (dx) a la posición x actual, y lo mismo para y (dy).
        self.snake_x += self.snake_dx
        self.snake_y += self.snake_dy

        # Por ahora, no nos preocupamos por los límites de la pantalla.
        # La serpiente simplemente se saldrá de la vista.
        # Esto lo gestionaremos en lecciones futuras.

def main():
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    arcade.run()

if __name__ == "__main__":
    main()

