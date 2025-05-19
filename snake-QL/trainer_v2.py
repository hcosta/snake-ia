# trainer_v1.py
import arcade
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt

from snake_v9 import SnakeGame, SCREEN_WIDTH, SCREEN_HEIGHT, STEP_SIZE, REWARD_STUCK_LOOP, POINTS_PER_FOOD
from agent_v1 import Agent

# --- Constantes de Entrenamiento ---
POSSIBLE_ACTIONS = [0, 1, 2, 3]

# Hiperparámetros del Agente
# Estos valores controlan cómo el agente aprende de sus experiencias.
# Ajustarlos puede tener un gran impacto en la velocidad de aprendizaje y
# el rendimiento final del agente.

LEARNING_RATE = 0.15
# LEARNING_RATE (Tasa de Aprendizaje, a menudo llamado 'alpha'):
# Define qué tan rápido el agente actualiza sus estimaciones de valor Q
# basadas en la nueva información (recompensa y valor Q futuro).
# - Un valor ALTO (cercano a 1): El agente da mucha importancia a la información más reciente,
#   aprendiendo rápido. Sin embargo, puede hacer que los valores Q oscilen mucho y
#   no converjan a la solución óptima, o que "olvide" lo aprendido previamente.
#   Podría ser útil si el entorno cambia mucho o si quieres una convergencia rápida inicial.
# - Un valor BAJO (cercano a 0): El agente aprende más lentamente, dando menos peso
#   a la información nueva y más a lo que ya sabía. Esto puede llevar a una
#   convergencia más estable y precisa, pero requiere más tiempo de entrenamiento.
# - Valor típico: 0.01 a 0.1. Tu valor de 0.1 es bastante estándar y bueno para empezar.
#   Si ves que el Avg Score es muy inestable o no mejora, podrías probar a bajarlo (ej. 0.05, 0.01).
#   Si el aprendizaje es muy lento, podrías probar a subirlo un poco (ej. 0.15, 0.2), pero con cuidado.

DISCOUNT_FACTOR = 0.925
# DISCOUNT_FACTOR (Factor de Descuento, a menudo llamado 'gamma'):
# Determina la importancia de las recompensas futuras.
# - Un valor ALTO (cercano a 1): El agente se preocupa mucho por las recompensas a largo plazo.
#   Es "visionario" y considerará el impacto de sus acciones actuales en recompensas
#   lejanas en el futuro. Esto es crucial para juegos como Snake, donde una
#   decisión temprana puede llevar a una trampa mucho después.
# - Un valor BAJO (cercano a 0): El agente es "miope" o "impaciente", priorizando las
#   recompensas inmediatas sobre las futuras. Esto puede ser útil en entornos donde
#   las acciones tienen consecuencias principalmente a corto plazo.
# - Valor típico: 0.9 a 0.99 para problemas que requieren planificación a largo plazo.
#   Tu valor de 0.9 es bueno. Si quisieras que el agente planifique aún más a largo plazo
#   (intentando evitar trampas muy lejanas), podrías probar con 0.95 o incluso 0.99.
#   Sin embargo, valores muy altos pueden hacer que el aprendizaje sea más lento en propagar
#   la información de recompensa hacia atrás.

EPSILON_START = 1.0
# EPSILON_START (Valor Inicial de Epsilon):
# Epsilon es la probabilidad de que el agente tome una acción aleatoria (exploración)
# en lugar de la mejor acción conocida según su tabla Q (explotación).
# - Empezar con epsilon = 1.0 significa que el agente comienza explorando completamente al azar.
#   Esto es bueno porque al principio no sabe nada y necesita descubrir qué acciones
#   llevan a buenas recompensas en diferentes estados.
# - Valor típico: 1.0. No suele modificarse mucho.

EPSILON_DECAY = 0.9995
# EPSILON_DECAY (Tasa de Decaimiento de Epsilon):
# Después de cada episodio (o a veces cada paso), epsilon se multiplica por este valor.
# Esto hace que la probabilidad de exploración disminuya gradualmente con el tiempo,
# permitiendo que el agente explote más el conocimiento que ha adquirido.
# - Un valor MÁS CERCANO A 1 (ej. 0.9999): Epsilon decae más LENTAMENTE.
#   El agente pasará más tiempo explorando. Puede ser bueno si el espacio de estados
#   es muy grande o si hay muchas soluciones subóptimas en las que el agente podría
#   atascarse si deja de explorar demasiado pronto.
# - Un valor MÁS BAJO (ej. 0.999 o 0.995): Epsilon decae MÁS RÁPIDO.
#   El agente pasará a explotar su conocimiento antes. Puede ser bueno si quieres
#   una convergencia más rápida a una buena política, pero arriesgas que no explore lo suficiente.
# - Tu valor de 0.9999 hace que epsilon decaiga de forma relativamente lenta, lo cual es
#   generalmente bueno para Snake. Tus logs muestran que epsilon llega a 0.0000 (probablemente
#   ha alcanzado EPSILON_MIN) después de 19000 episodios, lo cual es razonable.
#   Si sientes que el agente se estanca porque no explora lo suficiente en etapas tardías
#   (aunque con epsilon ya en su mínimo, esto depende de EPSILON_MIN), podrías hacer que
#   decaiga aún más lento (ej. 0.99995), o más rápido si crees que explora demasiado
#   tiempo (ej. 0.9995).

EPSILON_MIN = 0.0000001
# EPSILON_MIN (Valor Mínimo de Epsilon):
# Es el valor mínimo al que epsilon puede decaer.
# Incluso cuando el agente ha aprendido mucho, es útil mantener un pequeño nivel
# de exploración para poder adaptarse si el entorno cambia o para escapar de
# óptimos locales que no son la mejor solución global.
# - Un valor MUY BAJO (como el tuyo, 0.000001, o incluso 0.0): El agente eventualmente
#   dejará de explorar casi por completo (o del todo si es 0) y solo explotará
#   la política aprendida. Esto es bueno para evaluar el rendimiento final de la política.
# - Un valor MÁS ALTO (ej. 0.01, 0.05): El agente siempre mantendrá un nivel
#   significativo de exploración. Podría ser útil si sospechas que el agente
#   se queda atascado en políticas subóptimas.
# - Tu valor actual es muy bajo, lo que significa que al final del entrenamiento, el agente
#   está jugando de forma casi completamente determinista según su tabla Q.
#   Esto es bueno para ver qué tan bien ha aprendido. Si ves que se estanca
#   y sospechas que es por falta de exploración para salir de un "valle" de la función de valor,
#   podrías considerar aumentarlo ligeramente (ej. a 0.001 o 0.01) durante una fase de
#   entrenamiento, aunque para la "producción" o evaluación final, un epsilon mínimo bajo es común.
#   Dado que tus scores son altos, un epsilon mínimo muy bajo como el actual parece adecuado
#   para la fase de explotación.

Q_TABLE_FILENAME = "snake_q_table_v2.pkl"

# Parámetros de Entrenamiento
NUM_EPISODES_DEFAULT = 5000

# Control de Visualización
VISUALIZE_TRAINING_DEFAULT = True
VISUALIZATION_UPDATE_RATE = 0.03
SAVE_Q_TABLE_EVERY = 1000
PRINT_STATS_EVERY = 100

# --- Matplotlib Setup ---
plt.ion()
fig_plot, ax_plot = None, None
plot_scores = []
plot_avg_scores = []


def setup_matplotlib_plot():
    global fig_plot, ax_plot
    fig_plot, ax_plot = plt.subplots(figsize=(10, 5))
    ax_plot.set_title("Resultados del Entrenamiento de Snake IA")
    ax_plot.set_xlabel("Episodio")
    ax_plot.set_ylabel("Puntuación")
    line_score, = ax_plot.plot([], [], 'b-', label='Puntuación Episodio')
    line_avg_score, = ax_plot.plot(
        [], [], 'r-', label='Media Puntuación')
    ax_plot.legend()
    return line_score, line_avg_score


def update_matplotlib_plot(episode_num, new_score, scores_history_for_plot, line_score, line_avg_score):
    episodes_x = list(range(1, len(scores_history_for_plot) + 1))
    line_score.set_data(episodes_x, scores_history_for_plot)

    current_avg_scores_for_plot = []
    if scores_history_for_plot:
        for i in range(len(scores_history_for_plot)):
            current_avg_scores_for_plot.append(
                np.mean(scores_history_for_plot[max(0, i-99):i+1]))

    if current_avg_scores_for_plot:
        line_avg_score.set_data(episodes_x, current_avg_scores_for_plot)

    ax_plot.relim()
    ax_plot.autoscale_view(True, True, True)
    fig_plot.canvas.draw()
    plt.pause(0.001)


def run_ai_training(num_episodes, visualize_training, visualization_update_rate, reset_q_table_flag):
    global plot_scores, plot_avg_scores
    plot_scores = []
    plot_avg_scores = []

    env = None  # Inicializar env a None

    if reset_q_table_flag:
        if os.path.exists(Q_TABLE_FILENAME):
            try:
                os.remove(Q_TABLE_FILENAME)
                print(
                    f"Se ha eliminado '{Q_TABLE_FILENAME}' para reiniciar el entrenamiento.")
            except OSError as e:
                print(f"Error al eliminar '{Q_TABLE_FILENAME}': {e}")
        else:
            print(
                f"No se encontró '{Q_TABLE_FILENAME}'. Se iniciará un nuevo entrenamiento desde cero.")

    env_title = "Snake IA - Entrenamiento (Pulsa ESC o cierra la ventana para detener)"
    # Solo creamos la ventana si la visualización está activa
    if visualize_training:
        env = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT,
                        env_title, ai_controlled=True)
    else:
        # Si no hay visualización, necesitamos un 'dummy' env o adaptar la lógica.
        # Por simplicidad y dado que el error se da al visualizar, asumimos que
        # si no se visualiza, no se crea la ventana de Arcade.
        # Sin embargo, el juego DEBE existir para la lógica de la IA.
        # Creamos el entorno sin que sea una ventana visible si no se visualiza.
        # Esto requeriría que SnakeGame no herede de arcade.Window o tenga un modo no-GUI.
        # Para este caso, asumimos que SIEMPRE se crea el objeto SnakeGame,
        # pero solo se "usa" como ventana si visualize_training es True.
        # Esto es un poco complicado con la estructura actual.
        # La forma más directa es que 'env' SIEMPRE sea una instancia de SnakeGame.
        env = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT,
                        env_title, ai_controlled=True)

    agent = Agent(actions=POSSIBLE_ACTIONS,
                  learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                  epsilon=EPSILON_START,
                  epsilon_min=EPSILON_MIN, q_table_filepath=Q_TABLE_FILENAME)

    if reset_q_table_flag:
        agent.reset_epsilon()
        print(f"Epsilon reiniciado a: {agent.epsilon:.4f} debido a --reset.")

    scores_history_session = []
    total_steps = 0
    training_manually_stopped = False
    last_processed_episode = 0

    line_score, line_avg_score = None, None
    if visualize_training:
        line_score, line_avg_score = setup_matplotlib_plot()

    print(f"Iniciando entrenamiento para {num_episodes} episodios...")
    print(
        f"Visualización: {'Activada' if visualize_training else 'Desactivada'}")
    start_time_total = time.time()

    try:
        for episode_idx in range(1, num_episodes + 1):
            last_processed_episode = episode_idx
            # El reset del entorno se hace independientemente de si se visualiza
            current_state = env.reset()

            # La bandera training_should_stop se chequea desde el objeto env
            if env.training_should_stop:
                training_manually_stopped = True

            if training_manually_stopped:
                print(
                    f"Bucle de episodios detenido antes de iniciar episodio {episode_idx}.")
                break

            episode_score = 0
            episode_reward_sum = 0
            done = False
            steps_in_episode = 0
            steps_since_last_food = 0
            last_score_for_food_check = 0

            while not done:
                if visualize_training:
                    # dispatch_events solo es relevante si hay una ventana activa
                    env.dispatch_events()
                    if env.training_should_stop:  # Re-chequear después de eventos
                        training_manually_stopped = True

                if training_manually_stopped:
                    print(
                        f"Interrumpiendo pasos del episodio {episode_idx} (después de dispatch_events).")
                    break

                action = agent.get_action(current_state)
                next_state, reward, done, info = env.step(action)

                episode_reward_sum += reward

                if env.score > last_score_for_food_check:
                    steps_since_last_food = 0
                    last_score_for_food_check = env.score
                else:
                    steps_since_last_food += 1

                # max_steps_no_food = 75 + len(env.snake_body) * 5
                # Revisión, a ver si no queda atrapado dentro suyo
                # max_steps_no_food = 150 + len(env.snake_body) * 2

                # Revisión 2, Si no come en 60 pasos en un tablero tan pequeño
                # es muy probable que esté en una mala situación o en un bucle ineficiente.
                max_steps_no_food = 60

                if steps_since_last_food > max_steps_no_food:
                    reward += REWARD_STUCK_LOOP
                    done = True
                    env.game_over = True

                agent.train(current_state, action, reward, next_state, done)
                current_state = next_state
                episode_score = env.score
                steps_in_episode += 1
                total_steps += 1

                if visualize_training:
                    env.clear()
                    env.on_draw()
                    arcade.finish_render()
                    if done:
                        time.sleep(0.5)
                    else:
                        if visualization_update_rate > 0:  # Solo dormir si la tasa es positiva
                            time.sleep(visualization_update_rate)
                        # Si es 0, no se llama a time.sleep() o time.sleep(0) se ejecuta,
                        # lo cual es lo más rápido posible.

                if done:
                    break
            # --- Fin del bucle de pasos (while not done) ---

            if training_manually_stopped:
                break

            scores_history_session.append(episode_score)
            plot_scores.append(episode_score)
            agent.decay_epsilon()

            # MODIFICACIÓN: Actualizar el gráfico después de CADA episodio
            if visualize_training and line_score:
                update_matplotlib_plot(
                    episode_idx, episode_score, plot_scores, line_score, line_avg_score)

            if episode_idx % PRINT_STATS_EVERY == 0 or episode_idx == num_episodes:
                avg_score_last_100 = np.mean(
                    scores_history_session[-100:]) if scores_history_session else 0
                print(f"Ep: {episode_idx}/{num_episodes} | Steps: {steps_in_episode} | Score: {episode_score} | "
                      f"Total Reward: {episode_reward_sum:.1f} | Epsilon: {agent.epsilon:.4f} | "
                      f"Avg Score (actual): {avg_score_last_100:.2f}")
                if visualize_training and line_score:
                    update_matplotlib_plot(
                        episode_idx, episode_score, plot_scores, line_score, line_avg_score)

            if episode_idx % SAVE_Q_TABLE_EVERY == 0 or episode_idx == num_episodes or training_manually_stopped:
                agent.save_q_table(Q_TABLE_FILENAME)
        # --- Fin del bucle de episodios (for episode_idx...) ---

    except Exception as e:
        print(f"Ocurrió un error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time_total = time.time()
        if training_manually_stopped:
            completed_episodes_count = last_processed_episode - \
                1 if last_processed_episode > 0 else 0
            print(
                f"\nEntrenamiento detenido por el usuario. Se procesaron parcialmente hasta el episodio {last_processed_episode}.")
            if completed_episodes_count > 0:
                print(
                    f"Se completaron {completed_episodes_count} episodios en esta sesión.")
        else:
            completed_episodes_count = num_episodes if not training_manually_stopped else last_processed_episode
            print(
                f"\nEntrenamiento finalizado. Se completaron {completed_episodes_count} episodios en esta sesión.")

        print(
            f"Duración total de la sesión: {((end_time_total - start_time_total)/60):.2f} minutos.")
        print(f"Total de pasos ejecutados en la sesión: {total_steps}")

        agent.save_q_table(Q_TABLE_FILENAME)
        print(f"Tabla Q final (o de progreso) guardada en: {Q_TABLE_FILENAME}")

        # CORRECCIÓN AQUÍ:
        if env:  # Asegurarse que env no es None
            env.close()  # Usar el método close() de la instancia de la ventana

        if fig_plot:  # Solo cerrar si se creó
            plt.ioff()
            plt.close(fig_plot)


def play_with_trained_agent(q_table_path=Q_TABLE_FILENAME, num_episodes=5, visualization_speed=0.08):
    print(f"\n--- Viendo jugar al agente entrenado ({q_table_path}) ---")
    env_play = None  # Inicializar a None

    if not os.path.exists(q_table_path):
        print(
            f"Error: No se encontró el archivo de la Q-Table en '{q_table_path}'.")
        print("Por favor, entrena al agente primero o verifica la ruta.")
        return

    try:
        env_title = "Snake IA - Agente Entrenado (Pulsa ESC o cierra la ventana para salir)"
        env_play = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT,
                             env_title, ai_controlled=True)

        agent = Agent(actions=POSSIBLE_ACTIONS,
                      epsilon=0.001,
                      q_table_filepath=q_table_path)
        agent.epsilon_min = 0.0

        if not agent.q_table:
            # env_play podría no estar inicializado si la q_table no se carga,
            # pero en esta estructura, env_play se crea antes.
            # De todas formas, el return previene más ejecución.
            return

        for episode in range(1, num_episodes + 1):
            current_state = env_play.reset()
            env_play.training_should_stop = False
            done = False
            episode_score = 0
            print(f"\nIniciando demostración - Episodio {episode}")

            while not done:
                env_play.dispatch_events()
                if env_play.training_should_stop:
                    print(
                        "Saliendo de la demostración por petición del usuario (ESC o cierre de ventana).")
                    # No necesitamos cerrar aquí, el finally lo hará
                    return

                env_play.clear()
                env_play.on_draw()
                arcade.finish_render()
                time.sleep(visualization_speed)

                action = agent.get_action(current_state)
                next_state, reward, done, info = env_play.step(action)

                current_state = next_state
                episode_score = env_play.score

                if done:
                    print(
                        f"Episodio {episode} finalizado. Puntuación: {episode_score}")
                    env_play.clear()
                    env_play.on_draw()
                    arcade.finish_render()
                    time.sleep(2)
                    break

            if env_play.training_should_stop:
                break
    except Exception as e:
        print(f"Ocurrió un error durante la demostración: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # CORRECCIÓN AQUÍ:
        if env_play:  # Asegurarse que env_play no es None
            env_play.close()  # Usar el método close() de la instancia de la ventana


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenar o ver jugar a la IA de Snake.")
    parser.add_argument("--play", action="store_true",
                        help="Ver jugar al agente con la Q-Table guardada.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT,
                        help=f"Número de episodios para entrenar (default: {NUM_EPISODES_DEFAULT}).")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                        help="Desactivar la visualización del juego Arcade durante el entrenamiento.")
    parser.add_argument("--play-episodes", type=int, default=5,
                        help="Número de episodios para ver jugar al agente entrenado (default: 5).")
    parser.add_argument("--play-speed", type=float, default=0.08,
                        help="Segundos entre frames al ver jugar al agente (ej: 0.08).")
    parser.add_argument("--reset", action="store_true",
                        help="Reiniciar el entrenamiento eliminando la Q-Table guardada.")
    parser.add_argument("--visualization-speed", type=float, default=VISUALIZATION_UPDATE_RATE,
                        help=f"Segundos de pausa entre frames en la visualización del entrenamiento (ej: {VISUALIZATION_UPDATE_RATE}). Poner a 0 para máxima velocidad.")

    parser.set_defaults(visualize=VISUALIZE_TRAINING_DEFAULT)
    args = parser.parse_args()

    if args.play:
        play_with_trained_agent(q_table_path=Q_TABLE_FILENAME,
                                num_episodes=args.play_episodes,
                                visualization_speed=args.play_speed)
    else:
        effective_visualization_update_rate = args.visualization_speed
        # Nota: Si args.visualize es False, el objeto 'env' se crea igualmente
        # pero sus métodos de dibujo y dispatch_events no se usan intensivamente.
        # La lógica de on_close() en SnakeGame no se dispararía si no hay ventana visible
        # y no se llama a dispatch_events().
        # La detención con ESC se gestiona por la bandera training_should_stop, que
        # es más genérica y no depende estrictamente de la visualización.
        run_ai_training(num_episodes=args.episodes,
                        visualize_training=args.visualize,
                        visualization_update_rate=effective_visualization_update_rate,
                        reset_q_table_flag=args.reset)
