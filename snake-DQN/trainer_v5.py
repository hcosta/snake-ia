# trainer_v5.py

"""
Correcciones de la versión 5
============================
* Se ha modificado los scripts agent_v4.py y trainer_v4.py para usar el método recomendado de Keras para guardar y cargar modelos completos (model.save() y tf.keras.models.load_model())
* Se ha implementado numba y compilación jit para optimizar la lógica del juego y el estado del agente.
* Además se guarda el estado del entrenamiento para continuar desde el último episodio excepto si se hace un reset.
* La velocidad de entrenamiento ha mejorado más de 10 veces en comparación con la versión 4.
"""
import matplotlib as mpl
from agent_v5 import DQNAgent, AGENT_VERSION, DATA_DIR
from snake_logic_v11 import SnakeLogic, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC
import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
import pickle
from collections import deque

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if 1:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(
            f"Política de precisión mixta global: {mixed_precision.global_policy().name}")
    except Exception as e:
        print(f"No se pudo establecer mixed_float16: {e}. Usando float32.")

try:
    if 'DISPLAY' not in os.environ and os.name != 'nt':
        print("Configurando Matplotlib para backend 'Agg'.")
        mpl.use('Agg')
except ImportError:
    print("Advertencia: No se pudo importar matplotlib o establecer backend 'Agg'.")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu_dev in gpus:
            tf.config.experimental.set_memory_growth(gpu_dev, True)
        print(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error al habilitar crecimiento de memoria: {e}")
else:
    print("No se detectaron GPUs por TensorFlow.")

# --- Constantes y Hiperparámetros ---
STATE_SIZE = 11
ACTION_SIZE = 4
LEARNING_RATE_DQN_DEFAULT = 0.0001
DISCOUNT_FACTOR_DQN_DEFAULT = 0.99
EPSILON_START_DQN_DEFAULT = 1.0
EPSILON_DECAY_DQN_DEFAULT = 0.9995  # Es el epsilon_decay_rate para el agente
EPSILON_MIN_DQN_DEFAULT = 0.01
REPLAY_MEMORY_SIZE_DQN_DEFAULT = 1_000_000
BATCH_SIZE_DQN_DEFAULT = 4096
AGENT_EPOCHS_PER_REPLAY_DEFAULT = 1
UPDATE_TARGET_EVERY_DEFAULT = 10

MODEL_FILENAME_BASE_DEFAULT = "dqn_snake_checkpoint"  # Solo nombre base

NUM_EPISODES_DEFAULT = 100000
SAVE_CHECKPOINT_EVERY_DEFAULT = 500
PRINT_STATS_EVERY_DEFAULT = 10

plt.ion()  # type: ignore
fig_plot, ax_plot = None, None


def setup_matplotlib_plot():
    global fig_plot, ax_plot
    # type: ignore
    if fig_plot is None or not plt.fignum_exists(fig_plot.number):
        fig_plot, ax_plot = plt.subplots(figsize=(12, 6))
    else:
        ax_plot.clear()  # type: ignore
    ax_plot.set_title("Entrenamiento Snake IA (DQN)")  # type: ignore
    ax_plot.set_xlabel("Episodio (Global)")  # type: ignore
    ax_plot.set_ylabel("Puntuación")  # type: ignore
    line_score, = ax_plot.plot(
        [], [], 'b-', alpha=0.4, label='Puntuación Episodio')  # type: ignore
    line_avg_score, = ax_plot.plot(
        [], [], 'r-', linewidth=2, label='Media Puntuación (últ. 100)')  # type: ignore
    ax_plot.legend()  # type: ignore
    fig_plot.tight_layout()  # type: ignore
    return line_score, line_avg_score


def update_matplotlib_plot(plot_scores_data, line_score, line_avg_score):
    if not plot_scores_data or ax_plot is None or fig_plot is None:
        return  # type: ignore
    # item[0] es el número de episodio global
    episodes_x = [item[0] for item in plot_scores_data]
    scores_y = [item[1] for item in plot_scores_data]

    line_score.set_data(episodes_x, scores_y)
    if len(scores_y) >= 1:
        avg_scores_plot = [np.mean(scores_y[max(0, i-99):i+1])
                           for i in range(len(scores_y))]
        line_avg_score.set_data(episodes_x, avg_scores_plot)

    ax_plot.relim()  # type: ignore
    ax_plot.autoscale_view(True, True, True)  # type: ignore
    try:
        fig_plot.canvas.draw_idle()  # type: ignore
        fig_plot.canvas.flush_events()  # type: ignore
    except Exception as e:
        print(f"Error actualizando matplotlib: {e}")
    plt.pause(0.0001)


def run_ai_training(num_total_episodes_target, reset_model_flag, hyperparams, show_plot):
    plot_scores_session = []
    env = SnakeLogic(SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    model_base_name = hyperparams['model_file_base_name']

    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"Directorio de datos creado: {DATA_DIR}")
        except OSError as e:
            print(
                f"Error crítico al crear directorio {DATA_DIR}: {e}. Saliendo.")
            return

    model_keras_filepath = os.path.join(
        DATA_DIR, f"{model_base_name}_{AGENT_VERSION}.keras")
    training_state_filepath = os.path.join(
        DATA_DIR, f"{model_base_name}_{AGENT_VERSION}_train_state.pkl")

    start_episode = 1

    if reset_model_flag:
        print(
            f"--reset flag activado. Eliminando archivos en '{DATA_DIR}/' con base '{model_base_name}_{AGENT_VERSION}'...")
        if os.path.exists(model_keras_filepath):
            os.remove(model_keras_filepath)
            print(f"Modelo Keras eliminado: {model_keras_filepath}")
        if os.path.exists(training_state_filepath):
            os.remove(training_state_filepath)
            print(
                f"Archivo de estado de entrenamiento eliminado: {training_state_filepath}")

    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                     learning_rate=hyperparams['lr'], discount_factor=hyperparams['gamma'],
                     epsilon=hyperparams['initial_epsilon'],
                     # Corregido aquí
                     epsilon_decay_rate=hyperparams['epsilon_decay_rate'],
                     epsilon_min=EPSILON_MIN_DQN_DEFAULT,
                     replay_memory_size=hyperparams['replay_memory'],
                     batch_size=hyperparams['batch_size'],
                     model_filepath=model_keras_filepath,
                     epochs_per_replay=hyperparams['agent_epochs'])

    if not reset_model_flag and os.path.exists(training_state_filepath):
        print(
            f"Intentando cargar estado de entrenamiento desde {training_state_filepath}...")
        try:
            with open(training_state_filepath, 'rb') as f:
                training_state = pickle.load(f)
            start_episode = training_state.get('last_completed_episode', 0) + 1
            plot_scores_session = training_state.get('plot_scores_history', [])
            agent.epsilon = training_state.get(
                'agent_epsilon', agent.initial_epsilon)
            loaded_memory_list = training_state.get('agent_memory', [])
            agent.memory = deque(loaded_memory_list,
                                 maxlen=agent.replay_memory_capacity)
            print(
                f"Estado de entrenamiento cargado. Continuando desde episodio {start_episode}.")
            print(
                f"  Epsilon cargado: {agent.epsilon:.4f}, Tamaño de memoria: {len(agent.memory)}")
        except Exception as e:
            print(
                f"Error al cargar estado de entrenamiento desde {training_state_filepath}: {e}. Iniciando desde cero.")
            start_episode = 1
            plot_scores_session = []
            agent.reset_epsilon_and_memory(hyperparams['initial_epsilon'])
    elif not reset_model_flag:
        print(
            f"No se encontró {training_state_filepath}. Iniciando desde episodio 1.")
        start_episode = 1
        plot_scores_session = []
        agent.reset_epsilon_and_memory(hyperparams['initial_epsilon'])
    else:
        print("Iniciando entrenamiento desde episodio 1 debido a --reset.")
        start_episode = 1
        plot_scores_session = []
        agent.reset_epsilon_and_memory(hyperparams['initial_epsilon'])

    scores_history_for_avg = [s[1] for s in plot_scores_session if len(
        s) == 2]  # Asegurarse que tiene 2 elementos
    total_steps_session = 0
    training_manually_stopped = False
    line_score, line_avg_score = None, None  # type: ignore
    if show_plot:
        line_score, line_avg_score = setup_matplotlib_plot()
        if plot_scores_session:
            update_matplotlib_plot(plot_scores_session,
                                   line_score, line_avg_score)

    print(
        f"Iniciando/Continuando entrenamiento DQN hasta {num_total_episodes_target} episodios totales.")
    print(f"Comenzando esta sesión desde el episodio: {start_episode}")
    current_lr = tf.keras.backend.get_value(agent.model.optimizer.learning_rate) if hasattr(
        agent.model, 'optimizer') and agent.model.optimizer else agent.learning_rate
    print(f"Hiperparámetros: LR={current_lr:.6f}, Gamma={agent.gamma}, EpsilonActual={agent.epsilon:.4f}, "
          f"EpsilonDecayRate={hyperparams['epsilon_decay_rate']:.5f}, Batch={agent.batch_size}, EpochsPerReplay={agent.epochs_per_replay}, "
          f"ReplayMemCap={agent.memory.maxlen}, ModelFile='{agent.model_filepath}'")
    start_time_total_session = time.time()

    episode_idx_this_session = 0
    last_completed_episode_overall = start_episode - 1
    done_flag_for_finally = True

    try:
        for episode_actual_overall in range(start_episode, num_total_episodes_target + 1):
            last_completed_episode_overall = episode_actual_overall - 1
            episode_idx_this_session += 1
            current_state_tuple = env.reset()
            episode_reward_sum = 0.0
            done_flag_for_finally = False
            steps_in_episode = 0
            steps_since_last_food = 0
            max_steps_this_episode = (
                SCREEN_WIDTH_LOGIC // STEP_SIZE_LOGIC) * (SCREEN_HEIGHT_LOGIC // STEP_SIZE_LOGIC) * 2.5
            max_steps_this_episode = max(max_steps_this_episode, 100)

            for _ in range(int(max_steps_this_episode)):
                action_idx = agent.get_action(current_state_tuple)
                next_state_tuple, reward, done_flag_for_finally, info = env.step(
                    action_idx)
                episode_reward_sum += reward

                if info.get('ate_food', False):
                    steps_since_last_food = 0
                else:
                    steps_since_last_food += 1

                max_steps_no_food_limit = (
                    SCREEN_WIDTH_LOGIC + SCREEN_HEIGHT_LOGIC) // STEP_SIZE_LOGIC
                current_reward_for_memory = reward
                if not info.get('ate_food', False) and steps_since_last_food > max_steps_no_food_limit:
                    current_reward_for_memory -= 50
                    done_flag_for_finally = True

                agent.remember(current_state_tuple, action_idx,
                               current_reward_for_memory, next_state_tuple, done_flag_for_finally)
                current_state_tuple = next_state_tuple
                steps_in_episode += 1
                total_steps_session += 1

                if len(agent.memory) >= agent.batch_size and total_steps_session % 4 == 0:
                    agent.replay()

                if done_flag_for_finally:
                    break

            scores_history_for_avg.append(env.score)
            plot_scores_session.append((episode_actual_overall, env.score))

            if episode_actual_overall % UPDATE_TARGET_EVERY_DEFAULT == 0:
                agent.update_target_model()

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay_rate

            if episode_idx_this_session % PRINT_STATS_EVERY_DEFAULT == 0 or episode_actual_overall == num_total_episodes_target:
                avg_score_last_100 = np.mean(
                    scores_history_for_avg[-100:]) if scores_history_for_avg else 0.0
                print(f"Ep (Global): {episode_actual_overall}/{num_total_episodes_target} (Sesión: {episode_idx_this_session}) | "
                      f"Steps: {steps_in_episode} | Score: {env.score} | "
                      f"Total Reward Ep: {episode_reward_sum:.1f} | Epsilon: {agent.epsilon:.4f} | "
                      f"Avg Score (100ep): {avg_score_last_100:.2f} | Memory: {len(agent.memory)}")
                if show_plot and line_score is not None:
                    update_matplotlib_plot(
                        plot_scores_session, line_score, line_avg_score)  # type: ignore

            if episode_actual_overall % SAVE_CHECKPOINT_EVERY_DEFAULT == 0 or episode_actual_overall == num_total_episodes_target:
                agent.save_keras_model()  # Guarda en agent.model_filepath (que ya tiene DATA_DIR/)

                training_state_to_save = {
                    'last_completed_episode': episode_actual_overall,
                    'agent_epsilon': agent.epsilon,
                    'agent_memory': list(agent.memory),
                    'plot_scores_history': plot_scores_session
                }
                # training_state_filepath ya incluye DATA_DIR/
                with open(training_state_filepath, 'wb') as f:
                    pickle.dump(training_state_to_save, f)
                print(
                    f"Checkpoint (modelo Keras y estado de entrenamiento) guardado en episodio {episode_actual_overall}")

            last_completed_episode_overall = episode_actual_overall
    finally:
        end_time_total_session = time.time()
        print(
            f"\nSesión de entrenamiento {'detenida' if training_manually_stopped else 'finalizada'}.")
        print(
            f"Episodios completados en ESTA SESIÓN: {episode_idx_this_session-1 if training_manually_stopped and not done_flag_for_finally else episode_idx_this_session}")
        print(
            f"Último episodio global completado: {last_completed_episode_overall}")
        print(
            f"Duración de ESTA SESIÓN: {((end_time_total_session - start_time_total_session)/60):.2f} minutos.")
        print(
            f"Total de pasos ejecutados en ESTA SESIÓN: {total_steps_session}")

        # >=0 para permitir guardar incluso después del ep 0 si se interrumpe pronto
        if hasattr(agent, 'model') and agent.model and last_completed_episode_overall >= 0:
            print(
                f"Guardando estado final en episodio global {last_completed_episode_overall}...")
            agent.save_keras_model()

            final_training_state_to_save = {
                'last_completed_episode': last_completed_episode_overall,
                'agent_epsilon': agent.epsilon,
                'agent_memory': list(agent.memory),
                'plot_scores_history': plot_scores_session
            }
            try:
                with open(training_state_filepath, 'wb') as f:
                    pickle.dump(final_training_state_to_save, f)
                print(
                    f"Estado final del entrenamiento guardado en {training_state_filepath}.")
            except Exception as e_save:
                print(
                    f"Error al guardar el estado final del entrenamiento: {e_save}")

        if show_plot and fig_plot is not None:  # type: ignore
            plt.ioff()  # type: ignore
            try:
                if plt.fignum_exists(fig_plot.number):  # type: ignore
                    plot_save_filename = f"dqn_training_plot_{model_base_name}_{AGENT_VERSION}_ep{last_completed_episode_overall}.png"
                    plot_save_path = os.path.join(DATA_DIR, plot_save_filename)
                    plt.savefig(plot_save_path)  # type: ignore
                    print(
                        f"Gráfica de entrenamiento guardada como {plot_save_path}")
                    plt.close(fig_plot)  # type: ignore
            except Exception as e_plot:
                print(f"Error al guardar/cerrar la gráfica: {e_plot}")


def play_mode_with_ui(model_keras_path_full, num_episodes, play_speed):
    try:
        import arcade  # Importar solo aquí
        # Asumiendo que snake_ui_v10.py está disponible
        from snake_ui_v10 import SnakeGameUI
    except ImportError as e:
        print(
            f"Error al importar Arcade: {e}. Asegúrate de que está instalado para el modo play.")
        return

    print(
        f"\n--- Viendo jugar al agente DQN entrenado ({model_keras_path_full}) con UI ---")
    # model_keras_path_full ya viene con DATA_DIR/
    if not os.path.exists(model_keras_path_full):
        print(
            f"Error: No se encontró el archivo del modelo Keras en '{model_keras_path_full}'.")
        return

    game_logic = SnakeLogic(
        SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    agent_to_play = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                             epsilon=0.01, epsilon_min=0.01,
                             model_filepath=model_keras_path_full,
                             learning_rate=LEARNING_RATE_DQN_DEFAULT)

    ui_instance = None  # Para manejar el cierre
    try:
        ui_instance = SnakeGameUI(game_logic, agent_instance=agent_to_play,
                                  num_demo_episodes=num_episodes, play_speed=play_speed,
                                  ai_controlled_demo=True)
        arcade.run()
    except Exception as e:
        print(f"Error durante Arcade en modo play: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Demostración finalizada.")
        if ui_instance:
            arcade.exit()  # Intenta cerrar Arcade limpiamente
            # ui_instance.close() # close() podría no ser necesario si arcade.exit() funciona


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Entrenar o ver jugar IA de Snake. Datos en ./{DATA_DIR}/")
    parser.add_argument("--play", action="store_true",
                        help="Ver jugar al agente.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT,
                        help=f"Número TOTAL de episodios objetivo para el entrenamiento.")
    parser.add_argument("--plot", action="store_true",
                        help="Mostrar gráfica Matplotlib.")
    parser.add_argument("--play-episodes", type=int,
                        default=5, help="Episodios para ver jugar.")
    parser.add_argument("--play-speed", type=float,
                        default=0.05, help="Velocidad de juego de la IA.")
    parser.add_argument("--reset", action="store_true",
                        help=f"Reiniciar entrenamiento eliminando modelo y checkpoint de '{DATA_DIR}/'.")

    parser.add_argument("--lr", type=float, default=LEARNING_RATE_DQN_DEFAULT)
    parser.add_argument("--gamma", type=float,
                        default=DISCOUNT_FACTOR_DQN_DEFAULT)
    parser.add_argument("--epsilon_start", type=float,
                        default=EPSILON_START_DQN_DEFAULT)
    parser.add_argument("--epsilon_decay_rate", type=float,
                        default=EPSILON_DECAY_DQN_DEFAULT)  # Corregido aquí
    parser.add_argument("--batch_size", type=int,
                        default=BATCH_SIZE_DQN_DEFAULT)
    parser.add_argument("--replay_memory", type=int,
                        default=REPLAY_MEMORY_SIZE_DQN_DEFAULT)
    parser.add_argument("--agent_epochs", type=int,
                        default=AGENT_EPOCHS_PER_REPLAY_DEFAULT)

    parser.add_argument("--model_file_base", type=str, default=MODEL_FILENAME_BASE_DEFAULT,
                        help=f"Nombre base para archivos (ej: {MODEL_FILENAME_BASE_DEFAULT}). Se guardarán en '{DATA_DIR}/'.")

    args = parser.parse_args()

    current_hyperparams = {
        'lr': args.lr, 'gamma': args.gamma,
        'initial_epsilon': args.epsilon_start,
        'epsilon_decay_rate': args.epsilon_decay_rate,  # Corregido aquí
        'batch_size': args.batch_size,
        'replay_memory': args.replay_memory,
        'agent_epochs': args.agent_epochs,
        'model_filename': os.path.join(DATA_DIR, f"{args.model_file_base}_{AGENT_VERSION}.keras"),
        'model_file_base_name': args.model_file_base
    }

    if args.play:
        play_mode_with_ui(model_keras_path_full=current_hyperparams['model_filename'],
                          num_episodes=args.play_episodes, play_speed=args.play_speed)
    else:
        run_ai_training(num_total_episodes_target=args.episodes,
                        reset_model_flag=args.reset,
                        hyperparams=current_hyperparams,
                        show_plot=args.plot)
