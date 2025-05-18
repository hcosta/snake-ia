# snake_shell_v10.py
import curses
import argparse
import time

# Importar la lógica del juego y las constantes necesarias
from snake_logic_v10 import SnakeLogic, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC
# ACTION_MAP_LOGIC también se importa si es necesario referenciarlo directamente,
# pero para el juego humano crearemos un mapeo específico para curses.

# Mapeo de teclas de curses a los índices de acción de SnakeLogic
# Recordar ACTION_MAP_LOGIC:
# 0: DOWN (dy positiva)
# 1: UP (dy negativa)
# 2: LEFT
# 3: RIGHT
ACTION_MAP_HUMAN_CURSES = {
    curses.KEY_UP:    1,  # Envía acción 1 a SnakeLogic.step()
    curses.KEY_DOWN:  0,  # Envía acción 0
    curses.KEY_LEFT:  2,  # Envía acción 2
    curses.KEY_RIGHT: 3,  # Envía acción 3
}


def draw_game_shell(stdscr, game_logic_instance, terminal_rows, terminal_cols):
    stdscr.clear()
    game_board_char_height = game_logic_instance.height // game_logic_instance.step_size
    # ANCHO EN CELDAS LÓGICAS, NO EN CARACTERES AÚN
    game_board_logic_width = game_logic_instance.width // game_logic_instance.step_size

    # Dibujar bordes '#' - cada celda lógica de borde ahora son dos caracteres ##
    # El tablero tendrá (game_board_char_height + 2) filas de caracteres
    # y (game_board_logic_width * 2 + 2) columnas de caracteres (aprox)

    for r_idx in range(game_board_char_height + 2):
        # Iterar sobre celdas lógicas del borde
        for c_logic_idx in range(game_board_logic_width + 2):
            char_pair_to_draw = "##" if r_idx == 0 or \
                r_idx == game_board_char_height + 1 or \
                c_logic_idx == 0 or \
                c_logic_idx == game_board_logic_width + 1 \
                else "  "  # Espacio vacío para el interior por ahora

            # Convertir c_logic_idx a posición de caracter inicial
            c_char_start_idx = c_logic_idx * 2
            if 0 <= r_idx < terminal_rows and 0 <= c_char_start_idx + 1 < terminal_cols:  # +1 para el segundo char
                try:
                    if char_pair_to_draw != "  ":  # Solo dibujar bordes
                        stdscr.addstr(r_idx, c_char_start_idx,
                                      char_pair_to_draw)
                except curses.error:
                    pass

    # Dibujar comida (dentro de los bordes, por eso el +1 a las coordenadas lógicas)
    # Y luego *2 para la coordenada de caracter horizontal
    food_char_y = game_logic_instance.food_y // game_logic_instance.step_size + 1
    food_char_x_start = (game_logic_instance.food_x //
                         game_logic_instance.step_size + 1) * 2
    if 0 <= food_char_y < terminal_rows and 0 <= food_char_x_start + 1 < terminal_cols:
        try:
            # Comida como dos asteriscos
            stdscr.addstr(food_char_y, food_char_x_start, "**")
        except curses.error:
            pass

    # Dibujar serpiente
    for i, segment in enumerate(game_logic_instance.snake_body):
        seg_char_y = segment['y'] // game_logic_instance.step_size + 1
        seg_char_x_start = (
            segment['x'] // game_logic_instance.step_size + 1) * 2
        char_pair_to_draw = "SS" if i == 0 else "ss"  # O "OO", "oo"
        if 0 <= seg_char_y < terminal_rows and 0 <= seg_char_x_start + 1 < terminal_cols:
            try:
                stdscr.addstr(seg_char_y, seg_char_x_start, char_pair_to_draw)
            except curses.error:
                pass

    score_text = f"Score: {game_logic_instance.score}"
    score_pos_r = game_board_char_height + 2
    # El score no necesita duplicarse
    if 0 <= score_pos_r < terminal_rows and 0 + len(score_text) < terminal_cols:
        try:
            # Empezar un poco más a la derecha
            stdscr.addstr(score_pos_r, 2, score_text)
        except curses.error:
            pass

    if game_logic_instance.game_over:
        game_over_msg = "GAME OVER! 'q' to quit, 'r' to restart."
        msg_r = game_board_char_height // 2 + 1
        # Calcular la posición c para centrar el mensaje, considerando el ancho real en caracteres
        msg_c = ((game_board_logic_width + 2) * 2 - len(game_over_msg)) // 2
        if msg_c < 0:
            msg_c = 0
        if 0 <= msg_r < terminal_rows and 0 <= msg_c < terminal_cols:
            try:
                stdscr.addstr(msg_r, msg_c, game_over_msg)
            except curses.error:
                pass
    stdscr.refresh()


def game_loop_shell_curses(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(120)  # Velocidad del juego (milisegundos por frame)

    game = SnakeLogic(SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    # Para el juego humano, la serpiente ya empieza moviéndose a la derecha por el `setup` de SnakeLogic
    # `game.snake_dx_step` y `game.snake_dy_step` ya están inicializados.
    # La `current_human_action` debe reflejar esta dirección inicial.
    # Coincide con el setup
    current_human_action = ACTION_MAP_HUMAN_CURSES[curses.KEY_RIGHT]

    term_rows, term_cols = stdscr.getmaxyx()
    min_req_rows = (SCREEN_HEIGHT_LOGIC // STEP_SIZE_LOGIC) + \
        3  # +2 bordes, +1 score
    min_req_cols = (SCREEN_WIDTH_LOGIC // STEP_SIZE_LOGIC) + 2  # +2 bordes

    if term_rows < min_req_rows or term_cols < min_req_cols:
        stdscr.clear()
        stdscr.addstr(0, 0, "Terminal is too small.")
        stdscr.addstr(
            1, 0, f"Required: {min_req_rows} rows, {min_req_cols} cols.")
        stdscr.addstr(2, 0, f"Available: {term_rows} rows, {term_cols} cols.")
        stdscr.addstr(4, 0, "Press any key to exit.")
        stdscr.nodelay(0)
        stdscr.getch()
        return

    paused = False

    while True:
        user_key = stdscr.getch()
        # Re-check por si la terminal cambió de tamaño
        term_rows, term_cols = stdscr.getmaxyx()

        if game.game_over:
            if user_key == ord('q'):
                break
            elif user_key == ord('r'):
                game.reset()  # SnakeLogic.reset() ya llama a setup()
                # Reset dirección
                current_human_action = ACTION_MAP_HUMAN_CURSES[curses.KEY_RIGHT]
                paused = False
        else:  # Juego en curso
            if user_key == ord('p'):  # Pausa
                paused = not paused
                if paused:
                    stdscr.nodelay(0)  # Esperar input para despausar
                    pause_msg = "PAUSED - Press 'p' to continue"
                    msg_r = (SCREEN_HEIGHT_LOGIC // STEP_SIZE_LOGIC + 2) // 2
                    msg_c = (
                        (SCREEN_WIDTH_LOGIC // STEP_SIZE_LOGIC + 2) - len(pause_msg)) // 2
                    if msg_c < 0:
                        msg_c = 0
                    try:
                        stdscr.addstr(msg_r, msg_c, pause_msg)
                    except:
                        pass
                    stdscr.refresh()
                else:
                    stdscr.nodelay(1)
                    stdscr.timeout(120)  # Restaurar timeout

            if paused:
                # Si está pausado y la tecla no es 'p' para despausar, no hacer nada más en el bucle de juego
                if user_key != ord('p') and user_key != -1:  # -1 es no input
                    continue  # Esperar a que despause con 'p'
                elif user_key == -1 and paused:  # Si no hay input y está pausado
                    continue

            if user_key in ACTION_MAP_HUMAN_CURSES:
                # Lógica para evitar que la nueva acción sea la opuesta directa a la actual
                # Esto es para la sensación de control humano. SnakeLogic.step() ya lo maneja internamente.
                potential_new_action = ACTION_MAP_HUMAN_CURSES[user_key]
                allow_action_change = True

                # Si la serpiente tiene más de 1 segmento y ya se está moviendo
                if len(game.snake_body) > 1 and not (game.snake_dx_step == 0 and game.snake_dy_step == 0):
                    # Comprobar si la acción potencial es opuesta a la acción actual que está en curso
                    if (current_human_action == 0 and potential_new_action == 1) or \
                       (current_human_action == 1 and potential_new_action == 0) or \
                       (current_human_action == 2 and potential_new_action == 3) or \
                       (current_human_action == 3 and potential_new_action == 2):
                        allow_action_change = False

                if allow_action_change:
                    current_human_action = potential_new_action

            if not paused:
                game.step(current_human_action)  # Usa el `step` de SnakeLogic

        if not paused:  # Solo dibujar si no estamos esperando a que se despause
            draw_game_shell(stdscr, game, term_rows, term_cols)

        if user_key == ord('q'):  # Permitir 'q' para salir en cualquier momento
            break


def main_shell():
    parser = argparse.ArgumentParser(description="Snake Game Shell Version")
    parser.add_argument("--display-shell", type=int, default=1,  # Default a 1 para este script
                        help="Set to 1 to display the game in the shell.")
    # Aquí podrían ir otros argumentos si este script hiciera más cosas

    args = parser.parse_args()

    if args.display_shell == 1:
        try:
            curses.wrapper(game_loop_shell_curses)
        except ImportError:
            print(
                "El módulo 'curses' (o 'windows-curses' en Windows) no está disponible.")
            print("Intenta: pip install windows-curses (si estás en Windows)")
        except curses.error as e:
            print(f"Error de Curses: {e}")
            print(
                "Asegúrate de que la terminal es compatible y tiene el tamaño adecuado.")
            print(
                "Puede que necesites una terminal más grande o ajustar las dimensiones del juego.")
    else:
        print("Para ejecutar la versión de terminal, no desactives --display-shell.")
        print("Este script está diseñado para la visualización en shell.")
        # Opcionalmente, podrías llamar aquí a tu lógica de entrenamiento si este script fuera un todo en uno.


if __name__ == "__main__":
    main_shell()
