# trainer_v4.py
from agent_v4 import DQNAgent  # Usar el nuevo agent_v4
from snake_logic_v10 import SnakeLogic, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC
from snake_ui_v10 import SnakeGameUI  # Para el modo play, importamos la UI

import arcade  # Necesario para arcade.exit() y arcade.run() en el modo play
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
# Desactivar warnings y mensajes innecesarios de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 1:
    import tensorflow as tf

# HABILITAR CRECIMIENTO DE MEMORIA PARA LA GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu_dev in gpus:  # Renombrado para evitar conflicto con bucle for
            tf.config.experimental.set_memory_growth(gpu_dev, True)
        print(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error al habilitar crecimiento de memoria: {e}")
else:
    print("No se detectaron GPUs por TensorFlow para el entrenamiento headless.")


# --- Constantes de Entrenamiento ---
STATE_SIZE = 9
ACTION_SIZE = 4

# --- Hiperparámetros del Agente DQN (Defaults, pueden ser sobrescritos por CLI) ---
# Tasa de aprendizaje más baja para mayor estabilidad
LEARNING_RATE_DQN_DEFAULT = 0.00025
DISCOUNT_FACTOR_DQN_DEFAULT = 0.99    # Mayor gamma para visión a largo plazo
EPSILON_START_DQN_DEFAULT = 1.0
EPSILON_DECAY_DQN_DEFAULT = 0.9995  # Decaimiento por episodio (más lento)
EPSILON_MIN_DQN_DEFAULT = 0.01
REPLAY_MEMORY_SIZE_DQN_DEFAULT = 50000  # Memoria más grande
BATCH_SIZE_DQN_DEFAULT = 1024          # Batch grande para GPU
# Actualizar target model cada 10 episodios
UPDATE_TARGET_EVERY_DEFAULT = 10
AGENT_EPOCHS_PER_REPLAY_DEFAULT = 10   # Entrenar 10 epochs por llamada a replay

MODEL_FILENAME_DEFAULT = "dqn_snake_headless_v4.weights.h5"  # Nuevo nombre

# --- Parámetros de Entrenamiento del Trainer ---
NUM_EPISODES_DEFAULT = 100000
SAVE_MODEL_EVERY_DEFAULT = 500
PRINT_STATS_EVERY_DEFAULT = 10

# --- Matplotlib Setup ---
plt.ion()
fig_plot, ax_plot = None, None
# No inicializamos line_score y line_avg_score globalmente aquí


def setup_matplotlib_plot():
    global fig_plot, ax_plot
    # Crear solo si no existe o está cerrada
    if fig_plot is None or not plt.fignum_exists(fig_plot.number):
        fig_plot, ax_plot = plt.subplots(figsize=(12, 6))
    else:
        ax_plot.clear()  # Limpiar ejes si la figura ya existe

    ax_plot.set_title("Entrenamiento Snake IA (DQN Headless)")
    ax_plot.set_xlabel("Episodio")
    ax_plot.set_ylabel("Puntuación")
    line_score, = ax_plot.plot(
        [], [], 'b-', alpha=0.4, label='Puntuación Episodio')
    line_avg_score, = ax_plot.plot(
        [], [], 'r-', linewidth=2, label='Media Puntuación (últ. 100)')
    ax_plot.legend()
    fig_plot.tight_layout()
    return line_score, line_avg_score


def update_matplotlib_plot(plot_scores_data, line_score, line_avg_score):
    if not plot_scores_data:
        return

    episodes_x = list(range(1, len(plot_scores_data) + 1))
    line_score.set_data(episodes_x, plot_scores_data)

    if len(plot_scores_data) >= 1:
        avg_scores_plot = [np.mean(plot_scores_data[max(0, i-99):i+1])
                           for i in range(len(plot_scores_data))]
        line_avg_score.set_data(episodes_x, avg_scores_plot)

    ax_plot.relim()
    ax_plot.autoscale_view(True, True, True)
    try:
        fig_plot.canvas.draw_idle()  # Más eficiente que draw()
        fig_plot.canvas.flush_events()
    except Exception as e:
        print(f"Error actualizando matplotlib: {e}")
    plt.pause(0.0001)


def run_ai_training(num_episodes, reset_model_flag, hyperparams, show_plot):
    plot_scores_session = []

    env = SnakeLogic(SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    # Aplicar hiperparámetros (ya vienen en el diccionario hyperparams)
    agent = DQNAgent(state_size=STATE_SIZE,
                     action_size=ACTION_SIZE,
                     learning_rate=hyperparams['lr'],
                     discount_factor=hyperparams['gamma'],
                     epsilon=hyperparams['initial_epsilon'],
                     epsilon_decay=hyperparams['epsilon_decay'],
                     epsilon_min=EPSILON_MIN_DQN_DEFAULT,  # Epsilon min no suele cambiarse tanto
                     replay_memory_size=hyperparams['replay_memory'],
                     batch_size=hyperparams['batch_size'],
                     model_filepath=hyperparams['model_filename'],
                     epochs_per_replay=hyperparams['agent_epochs'])

    if reset_model_flag:
        if os.path.exists(agent.model_filepath):  # Usa el filepath del agente
            try:
                os.remove(agent.model_filepath)
                print(
                    f"Se ha eliminado '{agent.model_filepath}' para reiniciar el entrenamiento.")
            except OSError as e:
                print(f"Error al eliminar '{agent.model_filepath}': {e}")
        # Asegurar reseteo de epsilon
        agent.epsilon = hyperparams['initial_epsilon']
        print(f"Epsilon reiniciado a: {agent.epsilon:.4f} debido a --reset.")

    scores_history_session = []  # Para el avg score
    total_steps_session = 0
    training_manually_stopped = False

    line_score, line_avg_score = None, None
    if show_plot:
        line_score, line_avg_score = setup_matplotlib_plot()

    print(
        f"Iniciando entrenamiento DQN (Headless) para {num_episodes} episodios...")
    print(f"Hiperparámetros: LR={agent.learning_rate}, Gamma={agent.gamma}, EpsilonStart={agent.epsilon}, "
          f"EpsilonDecay={hyperparams['epsilon_decay']}, Batch={agent.batch_size}, EpochsPerReplay={agent.epochs_per_replay}, "
          f"ReplayMem={agent.memory.maxlen}")
    start_time_total = time.time()
    episode_idx_final = 0  # Para el finally

    try:
        for episode_idx in range(1, num_episodes + 1):
            episode_idx_final = episode_idx  # Guardar el último índice
            current_state_tuple = env.reset()
            # El agente espera la tupla, su remember/get_action se encarga de convertir/aplanar

            episode_reward_sum = 0
            done = False
            steps_in_episode = 0
            steps_since_last_food = 0

            # Ajustar límite de pasos por episodio si se desea
            max_steps_this_episode = (
                SCREEN_WIDTH_LOGIC // STEP_SIZE_LOGIC) * (SCREEN_HEIGHT_LOGIC // STEP_SIZE_LOGIC) * 2.5
            # Mínimo 100 pasos para no morir muy rápido
            max_steps_this_episode = max(max_steps_this_episode, 100)

            for _ in range(int(max_steps_this_episode)):  # Bucle de pasos con límite
                action_idx = agent.get_action(current_state_tuple)
                next_state_tuple, reward, done, info = env.step(action_idx)

                episode_reward_sum += reward

                if info.get('ate_food', False):
                    steps_since_last_food = 0
                else:
                    steps_since_last_food += 1

                max_steps_no_food_limit = (
                    SCREEN_WIDTH_LOGIC + SCREEN_HEIGHT_LOGIC) // STEP_SIZE_LOGIC  # Ajustado
                if not info.get('ate_food', False) and steps_since_last_food > max_steps_no_food_limit:
                    # REWARD_STUCK_LOOP (se puede definir como constante)
                    reward += -50
                    done = True
                    # info['collision_type'] = 'stuck_loop' # La lógica del env ya no tiene info

                agent.remember(current_state_tuple, action_idx,
                               reward, next_state_tuple, done)
                current_state_tuple = next_state_tuple
                steps_in_episode += 1
                total_steps_session += 1

                if len(agent.memory) > agent.batch_size:
                    # Frecuencia de replay() - ejemplo cada 4 pasos
                    if total_steps_session % 4 == 0:
                        agent.replay()

                if done:
                    break
            # --- Fin del bucle de pasos ---

            scores_history_session.append(env.score)
            plot_scores_session.append(env.score)

            if episode_idx % UPDATE_TARGET_EVERY_DEFAULT == 0:  # Usar constante global o pasada por hiperparámetro
                agent.update_target_model()

            if agent.epsilon > agent.epsilon_min:  # Decaimiento de Epsilon por episodio
                agent.epsilon *= hyperparams['epsilon_decay']

            if episode_idx % PRINT_STATS_EVERY_DEFAULT == 0 or episode_idx == num_episodes:
                avg_score_last_100 = np.mean(
                    scores_history_session[-100:]) if scores_history_session else 0.0
                print(f"Ep: {episode_idx}/{num_episodes} | Steps: {steps_in_episode} | Score: {env.score} | "
                      f"Total Reward: {episode_reward_sum:.1f} | Epsilon: {agent.epsilon:.4f} | "
                      f"Avg Score (last 100): {avg_score_last_100:.2f} | Memory: {len(agent.memory)}")
                if show_plot and line_score:  # Solo actualizar si se está mostrando la gráfica
                    update_matplotlib_plot(
                        plot_scores_session, line_score, line_avg_score)

            if episode_idx % SAVE_MODEL_EVERY_DEFAULT == 0 or episode_idx == num_episodes:
                agent.save_model(agent.model_filepath)
        # --- Fin del bucle de episodios ---

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
        training_manually_stopped = True
    except Exception as e:
        print(f"Ocurrió un error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time_total = time.time()
        completed_episodes_count = episode_idx_final
        # Si se detuvo a mitad de un episodio no contado
        if training_manually_stopped and episode_idx_final > 0 and not done:
            completed_episodes_count = episode_idx_final - 1

        print(
            f"\nEntrenamiento {'detenido' if training_manually_stopped else 'finalizado'}.")
        if completed_episodes_count > 0:
            print(
                f"Se completaron {completed_episodes_count} episodios en esta sesión.")
        print(
            f"Duración total de la sesión: {((end_time_total - start_time_total)/60):.2f} minutos.")
        print(f"Total de pasos ejecutados en la sesión: {total_steps_session}")

        if completed_episodes_count > 0:
            agent.save_model(agent.model_filepath)

        if show_plot and fig_plot:
            plt.ioff()
            try:
                if plt.fignum_exists(fig_plot.number):
                    plt.savefig("dqn_training_plot_headless_v4.png")
                    print(
                        "Gráfica de entrenamiento guardada como dqn_training_plot_headless_v4.png")
                    plt.close(fig_plot)
            except Exception as e_plot:
                print(f"Error al guardar/cerrar la gráfica: {e_plot}")


def play_mode_with_ui(model_path, num_episodes, play_speed):
    """Función para ver jugar al agente usando SnakeGameUI y SnakeLogic."""
    print(
        f"\n--- Viendo jugar al agente DQN entrenado ({model_path}) con UI ---")
    if not os.path.exists(model_path):
        print(
            f"Error: No se encontró el archivo del modelo en '{model_path}'. Entrena primero.")
        return

    game_logic = SnakeLogic(
        SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    # Cargar el agente entrenado. Epsilon bajo para explotación.
    # Usar los hiperparámetros guardados o unos por defecto para la estructura del modelo.
    # Es importante que state_size y action_size coincidan con el modelo guardado.
    agent_to_play = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                             epsilon=0.001, epsilon_min=0.001,
                             model_filepath=model_path)

    # Crear la instancia de la UI, pasándole la lógica, el agente, y los episodios a jugar
    SnakeGameUI(game_logic,
                agent_instance=agent_to_play,
                num_demo_episodes=num_episodes,
                play_speed=play_speed,
                ai_controlled_demo=True)
    try:
        arcade.run()  # Esto bloqueará hasta que la ventana de Arcade se cierre (después de num_demo_episodes)
    except Exception as e:
        print(f"Error durante la ejecución de Arcade en modo play: {e}")
    finally:
        print("Demostración finalizada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenar o ver jugar IA de Snake con DQN (Headless o UI).")
    parser.add_argument("--play", action="store_true",
                        help="Ver jugar al agente con el modelo guardado (usará UI).")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT,
                        help=f"Número de episodios para entrenar (default: {NUM_EPISODES_DEFAULT}).")
    parser.add_argument("--plot", action="store_true",
                        help="Mostrar gráfica Matplotlib durante entrenamiento headless.")
    parser.add_argument("--play-episodes", type=int, default=5,
                        help="Número de episodios para ver jugar (default: 5).")
    parser.add_argument("--play-speed", type=float, default=0.05,
                        help="Velocidad de la IA al jugar (segundos entre frames).")
    parser.add_argument("--reset", action="store_true",
                        help="Reiniciar entrenamiento eliminando el modelo guardado.")
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE_DQN_DEFAULT, help="Tasa de aprendizaje.")
    parser.add_argument("--gamma", type=float,
                        default=DISCOUNT_FACTOR_DQN_DEFAULT, help="Factor de descuento.")
    parser.add_argument("--epsilon_start", type=float,
                        default=EPSILON_START_DQN_DEFAULT, help="Epsilon inicial.")
    parser.add_argument("--epsilon_decay", type=float, default=EPSILON_DECAY_DQN_DEFAULT,
                        help="Decaimiento de Epsilon (por episodio).")
    parser.add_argument("--batch_size", type=int,
                        default=BATCH_SIZE_DQN_DEFAULT, help="Tamaño del Lote.")
    parser.add_argument("--replay_memory", type=int,
                        default=REPLAY_MEMORY_SIZE_DQN_DEFAULT, help="Tamaño Memoria de Repetición.")
    parser.add_argument("--agent_epochs", type=int,
                        default=AGENT_EPOCHS_PER_REPLAY_DEFAULT, help="Epochs por llamada a replay().")
    parser.add_argument("--model_file", type=str,
                        default=MODEL_FILENAME_DEFAULT, help="Nombre del archivo del modelo.")

    args = parser.parse_args()

    current_hyperparams = {
        'lr': args.lr,
        'gamma': args.gamma,
        'initial_epsilon': args.epsilon_start,  # El agente lo usa para resetearse
        # El trainer lo usa para decaer el epsilon del agente
        'epsilon_decay': args.epsilon_decay,
        'batch_size': args.batch_size,
        'replay_memory': args.replay_memory,
        'agent_epochs': args.agent_epochs,
        'model_filename': args.model_file
    }
    # Actualizar el nombre global si se cambia por CLI
    MODEL_FILENAME_DEFAULT = args.model_file

    if args.play:
        play_mode_with_ui(model_path=MODEL_FILENAME_DEFAULT,
                          num_episodes=args.play_episodes,
                          play_speed=args.play_speed)
    else:
        run_ai_training(num_episodes=args.episodes,
                        reset_model_flag=args.reset,
                        hyperparams=current_hyperparams,
                        show_plot=args.plot)

# Para Entrenar (Headless, dentro del contenedor Docker, ya no necesitas xvfb-run):
# python3 trainer_v4.py --episodes 100000 --batch_size 1024 --agent_epochs 10 --gamma 0.99 --epsilon_decay 0.9995 --lr 0.00025 --reset --plot

# Para Ver Jugar (en Windows con tu entorno Pipenv, después de entrenar y tener el .weights.h5):
# pipenv run python trainer_v4.py --play --play-episodes 10 --model_file dqn_snake_headless_v4.weights.h5
