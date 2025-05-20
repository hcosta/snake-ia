# snake_ui_v12.py
import arcade
from snake_logic_v13 import SnakeLogic, STEP_SIZE_LOGIC, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC

# --- Constantes Visuales ---
UI_SCREEN_WIDTH = SCREEN_WIDTH_LOGIC
UI_SCREEN_HEIGHT = SCREEN_HEIGHT_LOGIC
UI_SNAKE_SIZE = STEP_SIZE_LOGIC
UI_FOOD_SIZE = STEP_SIZE_LOGIC
UI_SCREEN_TITLE_DEFAULT = "Snake IA - Demostración"

# --- Colores ---
BACKGROUND_COLOR_UI = arcade.color.BLACK_OLIVE  # Un color diferente para la UI
SNAKE_COLOR_UI = arcade.color.LIME_GREEN
FOOD_COLOR_UI = arcade.color.RED_DEVIL
GAMEOVER_COLOR_UI = arcade.color.LIGHT_GRAY
SCORE_COLOR_UI = arcade.color.WHITE


class SnakeGameUI(arcade.Window):
    def __init__(self, snake_logic_instance: SnakeLogic,
                 agent_instance: object = None,  # Agente para controlar la serpiente
                 num_demo_episodes: int = 1,    # Número de episodios a jugar en modo IA
                 play_speed: float = 0.05,      # Segundos entre movimientos para la IA
                 ai_controlled_demo: bool = False):

        super().__init__(UI_SCREEN_WIDTH, UI_SCREEN_HEIGHT,
                         UI_SCREEN_TITLE_DEFAULT, update_rate=1/60)  # update_rate para fluidez
        arcade.set_background_color(BACKGROUND_COLOR_UI)

        self.game_logic = snake_logic_instance
        self.agent = agent_instance  # El agente DQN entrenado
        self.ai_controlled_demo = ai_controlled_demo

        self.num_demo_episodes_to_play = num_demo_episodes
        self.current_demo_episode_count = 0
        self.play_speed = play_speed  # Tiempo entre pasos para la IA
        self.time_since_last_ai_move = 0.0

        # Para control humano si ai_controlled_demo es False
        self.human_action_dx = 0
        self.human_action_dy = 0
        self.time_per_move_human = 0.1  # Velocidad para juego humano
        self.movement_timer_human = 0.0

        self.current_state_tuple_for_agent = None  # Guardar el estado para el agente

        if self.ai_controlled_demo:
            self._start_new_demo_episode()
        else:
            self.setup_human_play()  # Configuración para juego humano

    def setup_human_play(self):
        """Configura o resetea el juego para control humano."""
        self.game_logic.reset()
        self.human_action_dx = 0  # Empieza quieto
        self.human_action_dy = 0
        # Aunque no lo use el humano, para consistencia
        self.current_state_tuple_for_agent = self.game_logic.get_state()

    def _start_new_demo_episode(self):
        """Inicia un nuevo episodio para la demostración de la IA."""
        self.current_demo_episode_count += 1
        if self.current_demo_episode_count > self.num_demo_episodes_to_play:
            print(
                f"Demostración de {self.num_demo_episodes_to_play} episodios completada.")
            arcade.exit()  # Terminar la aplicación Arcade
            return

        print(
            f"\nIniciando demostración DQN (UI) - Episodio {self.current_demo_episode_count}/{self.num_demo_episodes_to_play}")
        self.current_state_tuple_for_agent = self.game_logic.reset()
        self.time_since_last_ai_move = 0.0

    def on_draw(self):
        self.clear()
        arcade.draw_rectangle_filled(
            center_x=self.game_logic.food_x, center_y=self.game_logic.food_y,
            width=UI_FOOD_SIZE, height=UI_FOOD_SIZE, color=FOOD_COLOR_UI
        )
        for segment in self.game_logic.snake_body:
            arcade.draw_rectangle_filled(
                center_x=segment['x'], center_y=segment['y'],
                width=UI_SNAKE_SIZE, height=UI_SNAKE_SIZE, color=SNAKE_COLOR_UI
            )
        score_text = f"{self.game_logic.score}"
        arcade.draw_text(score_text, 10, UI_SCREEN_HEIGHT -
                         30, SCORE_COLOR_UI, font_size=18)

        if self.ai_controlled_demo:
            episode_text = f"Ep: {self.current_demo_episode_count}/{self.num_demo_episodes_to_play}"
            arcade.draw_text(episode_text, UI_SCREEN_WIDTH - 10, UI_SCREEN_HEIGHT -
                             30, SCORE_COLOR_UI, font_size=18, anchor_x="right")

        if self.game_logic.game_over:
            arcade.draw_text("GAME OVER", UI_SCREEN_WIDTH / 2, (UI_SCREEN_HEIGHT / 2) + 50,
                             GAMEOVER_COLOR_UI, font_size=26, anchor_x="center", anchor_y="center")
            arcade.draw_text(f"{self.game_logic.score}", UI_SCREEN_WIDTH / 2, UI_SCREEN_HEIGHT / 2,
                             GAMEOVER_COLOR_UI, font_size=24, anchor_x="center", anchor_y="center")
            # Podrías obtener info['collision_type'] de la lógica si la guardas
            if not self.ai_controlled_demo:  # Solo para humano
                arcade.draw_text("ESPACIO = Reset", UI_SCREEN_WIDTH / 2, UI_SCREEN_HEIGHT / 2 - 40,
                                 GAMEOVER_COLOR_UI, font_size=16, anchor_x="center", anchor_y="center")

    def on_update(self, delta_time: float):
        if self.ai_controlled_demo:
            if self.game_logic.game_over:
                # La lógica para pasar al siguiente episodio o salir ya está en _start_new_demo_episode
                # y se llama de nuevo si es necesario, o arcade.exit() si se completaron.
                # Aquí podríamos añadir una pequeña pausa antes de que _start_new_demo_episode se llame
                # desde el siguiente ciclo de on_update si el juego acaba de terminar.
                # Por ahora, _start_new_demo_episode se llamará inmediatamente.
                self._start_new_demo_episode()
                if self.game_logic.game_over and self.current_demo_episode_count > self.num_demo_episodes_to_play:
                    arcade.exit()  # Asegurar salida
                    return
                return  # Salir de on_update si el juego terminó y se está reseteando o saliendo

            self.time_since_last_ai_move += delta_time
            if self.time_since_last_ai_move >= self.play_speed:
                self.time_since_last_ai_move = 0  # Resetear temporizador

                if self.agent and self.current_state_tuple_for_agent:
                    action_idx = self.agent.get_action(
                        self.current_state_tuple_for_agent)
                    next_state_tuple, _, done, info = self.game_logic.step(
                        action_idx)
                    self.current_state_tuple_for_agent = next_state_tuple

                    if done:
                        print(
                            f"Episodio {self.current_demo_episode_count} (IA) finalizado. Puntuación: {self.game_logic.score}. Colisión: {info.get('collision_type', 'N/A')}")
                        # La lógica para el siguiente episodio está arriba en el if self.game_logic.game_over
            return  # Fin de la lógica de IA para este update

        # Lógica para control Humano (si no es ai_controlled_demo)
        if self.game_logic.game_over:
            return

        self.movement_timer_human += delta_time
        if self.movement_timer_human >= self.time_per_move_human:
            self.movement_timer_human = 0  # Resetear temporizador

            action_to_take = None
            if self.human_action_dx == STEP_SIZE_LOGIC:
                action_to_take = 3  # RIGHT
            elif self.human_action_dx == -STEP_SIZE_LOGIC:
                action_to_take = 2  # LEFT
            elif self.human_action_dy == STEP_SIZE_LOGIC:
                action_to_take = 0  # UP
            elif self.human_action_dy == -STEP_SIZE_LOGIC:
                action_to_take = 1  # DOWN

            # Si está quieto al inicio
            if action_to_take is not None or (self.human_action_dx == 0 and self.human_action_dy == 0 and len(self.game_logic.snake_body) <= 3):
                # Permitir que el primer movimiento sea "nulo" para que la IA (o el humano) decida
                # O si es humano y no ha presionado tecla, step con la dirección actual
                # Si la acción es None (serpiente quieta), el step de la lógica lo manejará
                # Pero para humano, on_key_press establece la dirección

                # Para que el juego humano avance según la última tecla presionada:
                if self.human_action_dx != 0 or self.human_action_dy != 0:  # Solo si hay una dirección activa
                    # El step de la lógica usará snake_dx_step y snake_dy_step que on_key_press actualiza
                    # Necesitamos traducir esto a una acción para la interfaz step()
                    current_action_human = None
                    if self.game_logic.snake_dx_step > 0:
                        current_action_human = 3
                    elif self.game_logic.snake_dx_step < 0:
                        current_action_human = 2
                    elif self.game_logic.snake_dy_step > 0:
                        current_action_human = 0
                    elif self.game_logic.snake_dy_step < 0:
                        current_action_human = 1

                    if current_action_human is not None:
                        self.game_logic.step(current_action_human)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            arcade.exit()
            return

        if self.game_logic.game_over and not self.ai_controlled_demo:
            if key == arcade.key.SPACE:
                self.setup_human_play()  # Resetea para el jugador humano
            return

        if self.ai_controlled_demo:  # En modo demo IA, no procesar teclas de movimiento
            return

        # Control humano: actualiza la dirección en la instancia de game_logic
        # Estos se usan en on_update para el juego humano
        if key == arcade.key.UP:
            if self.game_logic.snake_dy_step != -STEP_SIZE_LOGIC:  # Evitar ir hacia atrás
                self.game_logic.snake_dy_step = STEP_SIZE_LOGIC
                self.game_logic.snake_dx_step = 0
        elif key == arcade.key.DOWN:
            if self.game_logic.snake_dy_step != STEP_SIZE_LOGIC:
                self.game_logic.snake_dy_step = -STEP_SIZE_LOGIC
                self.game_logic.snake_dx_step = 0
        elif key == arcade.key.LEFT:
            if self.game_logic.snake_dx_step != STEP_SIZE_LOGIC:
                self.game_logic.snake_dx_step = -STEP_SIZE_LOGIC
                self.game_logic.snake_dy_step = 0
        elif key == arcade.key.RIGHT:
            if self.game_logic.snake_dx_step != -STEP_SIZE_LOGIC:
                self.game_logic.snake_dx_step = STEP_SIZE_LOGIC
                self.game_logic.snake_dy_step = 0

        # Guardar la intención de movimiento para el on_update del humano
        self.human_action_dx = self.game_logic.snake_dx_step
        self.human_action_dy = self.game_logic.snake_dy_step

    def on_close(self):  # Se llama cuando se cierra la ventana de Arcade
        print("Ventana UI cerrada por el usuario.")
        super().on_close()
        arcade.exit()  # Asegurarse de que arcade se cierre bien


if __name__ == "__main__":
    game_logic = SnakeLogic(
        SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)
    game_ui = SnakeGameUI(game_logic, ai_controlled_demo=False)
    arcade.run()
