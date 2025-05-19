# snake_v1.py

import arcade

# --- Constantes de la Pantalla ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Snake IA - Aprendiendo Juntos v1"

class SnakeGame(arcade.Window):
    """
    Clase principal para nuestro juego del Snake.
    Hereda de arcade.Window para gestionar la ventana y los eventos del juego.
    """
    def __init__(self, width, height, title):
        # Llamamos al constructor de la clase padre (arcade.Window)
        # Esto configura la ventana con las dimensiones y el título dados.
        super().__init__(width, height, title)

        # Establecemos el color de fondo de la ventana.
        # Arcade tiene una lista de colores predefinidos en arcade.color
        # Un verde agradable para empezar
        arcade.set_background_color(arcade.color.AMAZON) 
    def on_draw(self):
        """
        Este método se llama cada vez que Arcade necesita (re)dibujar la ventana.
        Toda la lógica de dibujo debe ir aquí.
        """
        # self.clear() limpia la pantalla antes de dibujar el nuevo frame.
        # Es importante para evitar que los dibujos anteriores se queden "pegados".
        self.clear()

        # Por ahora, no dibujamos nada más que el fondo.
        # En futuras lecciones, aquí dibujaremos la serpiente, la comida, etc.

    def on_update(self, delta_time):
        """
        Este método se llama aproximadamente 60 veces por segundo (por defecto).
        Aquí va toda la lógica del juego que necesita actualizarse con el tiempo,
        como el movimiento, las colisiones, la puntuación, etc. delta_time es el
        tiempo en segundos que ha pasado desde la última llamada a on_update.
        """
        pass # De momento, nuestra lógica de juego no hace nada.

def main():
    """ Función principal para configurar y ejecutar el juego. """
    # Creamos una instancia de nuestra clase SnakeGame.
    game = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    # arcade.run() inicia el bucle principal del juego.
    # Mantiene la ventana abierta, maneja eventos y llama a on_draw y on_update.
    arcade.run()

# Esta es una construcción estándar en Python:
# El código dentro de este if solo se ejecuta si el script se corre directamente
# (no si se importa como un módulo en otro script).
if __name__ == "__main__":
    main()