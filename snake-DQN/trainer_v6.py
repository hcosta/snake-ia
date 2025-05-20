# trainer_v6.py

"""
Versión 6 del Trainer
=====================
* Utiliza agent_v6 y snake_logic_v12.
* STATE_SIZE es 12 para acomodar la nueva variable de estado.
* AGENT_VERSION es "6" para nombres de archivo.
"""
import matplotlib as mpl
# Asegurarse de que se importa DQNAgent de agent_v6 y SnakeLogic de snake_logic_v12
from agent_v6 import DQNAgent, AGENT_VERSION as AGENT_VERSION_FROM_AGENT, DATA_DIR
from snake_logic_v12 import SnakeLogic, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC

import numpy as np
import time
import os
import argparse
import matplotlib.pyplot as plt
import pickle
from collections import deque

# Optimización específica de CPU Intel, puede desactivarse
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suprime logs informativos de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuración de TensorFlow y GPU (si está disponible)
try:
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    # Configurar política de precisión mixta si se desea (para GPUs compatibles)
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # print(f"Política de precisión mixta global: {mixed_precision.global_policy().name}")
except Exception as e:
    print(
        f"Advertencia: No se pudo configurar mixed_precision: {e}. Usando float32 por defecto.")

try:
    # Configurar backend de Matplotlib para entornos sin GUI si es necesario
    if 'DISPLAY' not in os.environ and os.name != 'nt':  # Linux sin display
        print("Configurando Matplotlib para backend 'Agg'.")
        mpl.use('Agg')
except ImportError:
    print("Advertencia: No se pudo importar matplotlib o establecer backend 'Agg'.")


# Configuración de crecimiento de memoria para GPUs TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu_dev in gpus:
            tf.config.experimental.set_memory_growth(gpu_dev, True)
        print(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error al habilitar crecimiento de memoria para GPU: {e}")
else:
    print("No se detectaron GPUs por TensorFlow. Ejecutando en CPU.")


# --- Constantes y Hiperparámetros del Trainer ---
AGENT_VERSION_TRAINER = "6"  # Asegurar que coincide con la intención
if AGENT_VERSION_FROM_AGENT != AGENT_VERSION_TRAINER:
    print(
        f"Advertencia: AGENT_VERSION en trainer ({AGENT_VERSION_TRAINER}) y agente ({AGENT_VERSION_FROM_AGENT}) no coinciden.")
    # Podría ser crítico si los nombres de archivo dependen de esto de forma inconsistente.

STATE_SIZE = 12  # Actualizado para v6 (11 anteriores + 1 nueva)
# Número de acciones posibles (arriba, abajo, izquierda, derecha)
ACTION_SIZE = 4

# Valores por defecto para hiperparámetros (pueden ser sobrescritos por args)
LEARNING_RATE_DQN_DEFAULT = 0.0005  # Ajustado, era 0.0001
DISCOUNT_FACTOR_DQN_DEFAULT = 0.5
EPSILON_START_DQN_DEFAULT = 1.0
EPSILON_DECAY_DQN_DEFAULT = 0.9  # Más lento, era 0.9995
EPSILON_MIN_DQN_DEFAULT = 0.01
# Reducido para pruebas más rápidas, era 1M
REPLAY_MEMORY_SIZE_DQN_DEFAULT = 1_000_000
# Aumentado, era 4096, pero puede ser mucho para memoria pequeña. 64-512 es común.
BATCH_SIZE_DQN_DEFAULT = 16
# Cuántas epochs entrenar el modelo en cada llamada a replay()
AGENT_EPOCHS_PER_REPLAY_DEFAULT = 1
# Episodios entre actualizaciones del target_model
UPDATE_TARGET_EVERY_DEFAULT = 1

MODEL_FILENAME_BASE_DEFAULT = "dqn_snake_checkpoint"
NUM_EPISODES_DEFAULT = 10000
SAVE_CHECKPOINT_EVERY_DEFAULT = 500
PRINT_STATS_EVERY_DEFAULT = 10  # Episodios

# --- Configuración de Gráficas Matplotlib ---
plt.ion()  # Modo interactivo para Matplotlib
fig_plot, ax_plot = None, None


def setup_matplotlib_plot():
    global fig_plot, ax_plot
    # type: ignore
    if fig_plot is None or not plt.fignum_exists(fig_plot.number):
        fig_plot, ax_plot = plt.subplots(figsize=(12, 6))
    else:
        ax_plot.clear()  # type: ignore
    ax_plot.set_title(
        f"Entrenamiento Snake IA (DQN v{AGENT_VERSION_TRAINER})")  # type: ignore
    ax_plot.set_xlabel("Episodio (Global)")  # type: ignore
    ax_plot.set_ylabel("Puntuación")  # type: ignore
    # line_score, = ax_plot.plot([], [], 'b-', alpha=0.5, label='Puntuación Episodio') # type: ignore
    line_score, = ax_plot.plot([], [], marker='.', linestyle='-', color='b',
                               alpha=0.3, markersize=2, label='Puntuación Episodio')  # type: ignore
    line_avg_score, = ax_plot.plot(
        [], [], 'r-', linewidth=2, label='Media Puntuación (últ. 100)')  # type: ignore
    ax_plot.legend()  # type: ignore
    fig_plot.tight_layout()  # type: ignore
    return line_score, line_avg_score


def update_matplotlib_plot(plot_scores_data, line_score, line_avg_score):
    if not plot_scores_data or ax_plot is None or fig_plot is None:  # type: ignore
        return

    # item[0] es el número de episodio global
    episodes_x = [item[0] for item in plot_scores_data]
    scores_y = [item[1]
                for item in plot_scores_data]   # item[1] es la puntuación

    line_score.set_data(episodes_x, scores_y)

    if len(scores_y) >= 1:
        # Calcular media móvil de 100 episodios
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
    # Pausa muy breve para permitir que la gráfica se actualice
    plt.pause(0.0001)


def run_ai_training(num_total_episodes_target, reset_model_flag, hyperparams, show_plot):
    # Historial de puntuaciones para la gráfica (episodio_global, score)
    plot_scores_session = []
    env = SnakeLogic(SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    model_base_name = hyperparams['model_file_base_name']

    # Asegurar que el directorio de datos existe
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"Directorio de datos creado: {DATA_DIR}")
        except OSError as e:
            print(
                f"Error crítico al crear directorio {DATA_DIR}: {e}. Saliendo.")
            return

    # Definir paths para el modelo Keras y el estado del entrenamiento
    model_keras_filepath = os.path.join(
        DATA_DIR, f"{model_base_name}_{AGENT_VERSION_TRAINER}.keras")
    training_state_filepath = os.path.join(
        DATA_DIR, f"{model_base_name}_{AGENT_VERSION_TRAINER}_train_state.pkl")

    start_episode_global = 1  # Episodio global desde el que empezar

    if reset_model_flag:
        print(
            f"--reset flag activado. Eliminando archivos en '{DATA_DIR}/' con base '{model_base_name}_{AGENT_VERSION_TRAINER}'...")
        if os.path.exists(model_keras_filepath):
            try:
                os.remove(model_keras_filepath)
                print(f"Modelo Keras eliminado: {model_keras_filepath}")
            except Exception as e:
                print(f"Error eliminando modelo: {e}")
        if os.path.exists(training_state_filepath):
            try:
                os.remove(training_state_filepath)
                print(
                    f"Archivo de estado de entrenamiento eliminado: {training_state_filepath}")
            except Exception as e:
                print(f"Error eliminando estado: {e}")

    # Instanciar el agente
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                     learning_rate=hyperparams['lr'], discount_factor=hyperparams['gamma'],
                     epsilon=hyperparams['initial_epsilon'],
                     epsilon_decay_rate=hyperparams['epsilon_decay_rate'],
                     # Usar el valor de hyperparams
                     epsilon_min=hyperparams['epsilon_min'],
                     replay_memory_size=hyperparams['replay_memory'],
                     batch_size=hyperparams['batch_size'],
                     model_filepath=model_keras_filepath,  # Pasar el path completo
                     epochs_per_replay=hyperparams['agent_epochs'])

    # Cargar estado de entrenamiento si no se resetea y existe
    if not reset_model_flag and os.path.exists(training_state_filepath):
        print(
            f"Intentando cargar estado de entrenamiento desde {training_state_filepath}...")
        try:
            with open(training_state_filepath, 'rb') as f:
                training_state = pickle.load(f)
            start_episode_global = training_state.get(
                'last_completed_episode_overall', 0) + 1
            plot_scores_session = training_state.get('plot_scores_history', [])
            agent.epsilon = training_state.get(
                'agent_epsilon', agent.initial_epsilon)

            loaded_memory_list = training_state.get('agent_memory', [])
            # Reconstruir deque con maxlen correcto
            agent.memory = deque(loaded_memory_list,
                                 maxlen=agent.replay_memory_capacity)

            print(
                f"Estado de entrenamiento cargado. Continuando desde episodio global {start_episode_global}.")
            print(
                f"  Epsilon cargado: {agent.epsilon:.4f}, Tamaño de memoria: {len(agent.memory)}")
        except Exception as e:
            print(
                f"Error al cargar estado de entrenamiento desde {training_state_filepath}: {e}. Iniciando desde cero.")
            start_episode_global = 1
            plot_scores_session = []
            agent.reset_epsilon_and_memory(
                hyperparams['initial_epsilon'])  # Resetear agente
    elif not reset_model_flag:  # No reset, pero no hay archivo de estado
        print(
            f"No se encontró {training_state_filepath}. Iniciando desde episodio global 1.")
        start_episode_global = 1
        plot_scores_session = []
        agent.reset_epsilon_and_memory(hyperparams['initial_epsilon'])
    else:  # reset_model_flag es True
        print("Iniciando entrenamiento desde episodio global 1 debido a --reset.")
        start_episode_global = 1
        plot_scores_session = []
        agent.reset_epsilon_and_memory(hyperparams['initial_epsilon'])

    # Inicializar historial de puntuaciones para calcular la media
    scores_history_for_avg = deque(
        [s[1] for s in plot_scores_session], maxlen=100)

    total_steps_session = 0  # Pasos solo para esta sesión de ejecución
    training_manually_stopped = False

    line_score_plot, line_avg_score_plot = None, None
    if show_plot:
        line_score_plot, line_avg_score_plot = setup_matplotlib_plot()
        if plot_scores_session:  # Si se cargaron datos históricos, graficarlos
            update_matplotlib_plot(plot_scores_session,
                                   line_score_plot, line_avg_score_plot)

    print(
        f"Iniciando/Continuando entrenamiento DQN hasta {num_total_episodes_target} episodios globales.")
    print(
        f"Comenzando esta sesión desde el episodio global: {start_episode_global}")
    current_lr_val = tf.keras.backend.get_value(agent.model.optimizer.learning_rate) if hasattr(
        agent.model, 'optimizer') and agent.model.optimizer else agent.learning_rate
    print(f"Hiperparámetros: LR={current_lr_val:.6f}, Gamma={agent.gamma:.3f}, EpsilonActual={agent.epsilon:.4f}, "
          f"EpsilonDecayRate={hyperparams['epsilon_decay_rate']:.6f}, EpsilonMin={hyperparams['epsilon_min']:.4f}, Batch={agent.batch_size}, "
          f"EpochsPerReplay={agent.epochs_per_replay}, ReplayMemCap={agent.memory.maxlen}, "
          f"UpdateTargetEvery={hyperparams['update_target_every']} eps, ModelFile='{agent.model_filepath}'")

    start_time_total_session = time.time()

    # episode_idx_this_session: contador de episodios en *esta ejecución* del script
    # episode_actual_overall: número de episodio *global* (considerando ejecuciones previas)

    last_completed_episode_overall = start_episode_global - 1

    try:
        for episode_idx_this_session, episode_actual_overall in \
                enumerate(range(start_episode_global, num_total_episodes_target + 1), start=1):

            last_completed_episode_overall = episode_actual_overall - \
                1  # Lo que se completó antes de este
            current_state_tuple = env.reset()
            episode_reward_sum = 0.0
            done_episode = False
            steps_in_episode = 0

            # Límite de pasos por episodio para evitar bucles infinitos no productivos
            max_steps_per_episode = (SCREEN_WIDTH_LOGIC // STEP_SIZE_LOGIC) * \
                                    (SCREEN_HEIGHT_LOGIC // STEP_SIZE_LOGIC) * \
                2.5  # Unas vueltas al tablero
            max_steps_per_episode = max(
                max_steps_per_episode, 150)  # Mínimo de pasos

            for step_num_in_ep in range(int(max_steps_per_episode)):
                action_idx = agent.get_action(current_state_tuple)
                next_state_tuple, reward, done_episode, info = env.step(
                    action_idx)

                episode_reward_sum += reward

                # Guardar en memoria de repetición
                agent.remember(current_state_tuple, action_idx,
                               reward, next_state_tuple, done_episode)

                current_state_tuple = next_state_tuple
                steps_in_episode += 1
                total_steps_session += 1

                # Realizar un paso de entrenamiento (replay)
                # Condición: suficiente memoria y, opcionalmente, cada N pasos o cada episodio
                # Entrenar cada 4 pasos globales
                if len(agent.memory) >= agent.batch_size and total_steps_session % 4 == 0:
                    agent.replay()

                if done_episode:
                    break  # Terminar el episodio actual

            # Fin del episodio
            # Añadir puntuación para la media
            scores_history_for_avg.append(env.score)
            # Añadir para la gráfica completa
            plot_scores_session.append((episode_actual_overall, env.score))

            # Actualizar el modelo objetivo periódicamente
            if episode_actual_overall % hyperparams['update_target_every'] == 0:
                agent.update_target_model()
                # print(f"Target model updated at episode {episode_actual_overall}")

            # Decaimiento de Epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay_rate
                # Asegurar que no baja de epsilon_min
                agent.epsilon = max(agent.epsilon_min, agent.epsilon)

            # Imprimir estadísticas y actualizar gráfica
            if episode_idx_this_session % PRINT_STATS_EVERY_DEFAULT == 0 or episode_actual_overall == num_total_episodes_target:
                avg_score_last_100 = np.mean(list(scores_history_for_avg))
                print(f"Ep (Global): {episode_actual_overall}/{num_total_episodes_target} (Sesión: {episode_idx_this_session}) | "
                      f"Steps: {steps_in_episode:3} | Score: {env.score:3} | "
                      f"Total Reward Ep: {episode_reward_sum:6.1f} | Epsilon: {agent.epsilon:.4f} | "
                      f"Avg Score (100ep): {avg_score_last_100:6.2f} | Memory: {len(agent.memory):6}")
                if show_plot and line_score_plot is not None and line_avg_score_plot is not None:
                    update_matplotlib_plot(
                        plot_scores_session, line_score_plot, line_avg_score_plot)

            # Guardar checkpoint
            if episode_actual_overall % SAVE_CHECKPOINT_EVERY_DEFAULT == 0 or episode_actual_overall == num_total_episodes_target:
                agent.save_keras_model()  # Guarda modelo Keras

                training_state_to_save = {
                    'last_completed_episode_overall': episode_actual_overall,
                    'agent_epsilon': agent.epsilon,
                    # Guardar memoria como lista
                    'agent_memory': list(agent.memory),
                    'plot_scores_history': plot_scores_session
                }
                with open(training_state_filepath, 'wb') as f:
                    pickle.dump(training_state_to_save, f)
                print(
                    f"Checkpoint (modelo y estado) guardado en episodio global {episode_actual_overall}")

            # Actualizar el último completado
            last_completed_episode_overall = episode_actual_overall

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente por el usuario.")
        training_manually_stopped = True
    finally:
        end_time_total_session = time.time()
        print(
            f"\nSesión de entrenamiento {'detenida' if training_manually_stopped else 'finalizada'}.")

        # El número de episodios completados en *esta sesión*
        # Si se detuvo a mitad de un episodio, episode_idx_this_session podría ser 1 más de los completados.
        # last_completed_episode_overall tiene el último episodio global que *finalizó*.
        # episodios_en_esta_sesion = last_completed_episode_overall - (start_episode_global -1)

        # O más simple: si se detuvo manualmente Y el último episodio no terminó (done_episode es False)
        # entonces el número de episodios de la sesión es episode_idx_this_session - 1.
        # Pero es más fácil usar last_completed_episode_overall.

        num_eps_this_run = last_completed_episode_overall - \
            (start_episode_global - 1)

        print(f"Episodios completados en ESTA SESIÓN: {num_eps_this_run}")
        print(
            f"Último episodio global completado: {last_completed_episode_overall}")
        print(
            f"Duración de ESTA SESIÓN: {((end_time_total_session - start_time_total_session)/60):.2f} minutos.")
        print(
            f"Total de pasos ejecutados en ESTA SESIÓN: {total_steps_session}")

        # Guardar estado final (incluso si se interrumpe)
        if hasattr(agent, 'model') and agent.model and last_completed_episode_overall >= 0:
            print(
                f"Guardando estado final en episodio global {last_completed_episode_overall}...")
            agent.save_keras_model()  # Guardar modelo

            final_training_state_to_save = {
                'last_completed_episode_overall': last_completed_episode_overall,
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

        # Guardar y cerrar la gráfica
        if show_plot and fig_plot is not None:
            plt.ioff()  # type: ignore
            try:
                if plt.fignum_exists(fig_plot.number):  # type: ignore
                    plot_save_filename = f"dqn_training_plot_{model_base_name}_{AGENT_VERSION_TRAINER}_ep{last_completed_episode_overall}.png"
                    plot_save_path = os.path.join(DATA_DIR, plot_save_filename)
                    plt.savefig(plot_save_path)  # type: ignore
                    print(
                        f"Gráfica de entrenamiento guardada como {plot_save_path}")
                    plt.close(fig_plot)  # type: ignore
            except Exception as e_plot:
                print(f"Error al guardar/cerrar la gráfica: {e_plot}")


def play_mode_with_ui(model_keras_path_full, num_episodes_to_play, play_speed_ui):
    """Ejecuta el agente en modo demostración con UI de Arcade."""
    try:
        import arcade
        # Asumimos que snake_ui_v10.py está disponible y que su SnakeGameUI puede tomar cualquier SnakeLogic
        # UI no cambia, solo la lógica que se le pasa.
        from snake_ui_v11 import SnakeGameUI
    except ImportError as e:
        print(
            f"Error al importar Arcade o SnakeGameUI: {e}. Asegúrate de que Arcade está instalado.")
        print("La UI usa 'snake_ui_v10.py'. Asegúrate de que es compatible o actualízala si es necesario.")
        return

    print(
        f"\n--- Viendo jugar al agente DQN entrenado ({model_keras_path_full}) con UI ---")
    if not os.path.exists(model_keras_path_full):
        print(
            f"Error: No se encontró el archivo del modelo Keras en '{model_keras_path_full}'.")
        print("Asegúrate de que el modelo ha sido entrenado y guardado con la versión correcta del agente.")
        return

    # Usar la nueva lógica para la demostración también
    game_logic_instance_for_play = SnakeLogic(
        SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    # Agente para jugar (epsilon muy bajo para explotación)
    # El state_size debe ser el que espera el modelo cargado (12 para v6)
    agent_for_playing = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                                 epsilon=0.0000001, epsilon_min=0.0000001,  # Epsilon muy bajo para jugar
                                 model_filepath=model_keras_path_full,
                                 learning_rate=0.0000001)  # LR no es necesario para jugar

    ui_instance = None
    try:
        ui_instance = SnakeGameUI(
            snake_logic_instance=game_logic_instance_for_play,
            agent_instance=agent_for_playing,
            num_demo_episodes=num_episodes_to_play,
            play_speed=play_speed_ui,
            ai_controlled_demo=True
        )
        arcade.run()
    except Exception as e:
        print(f"Error durante la ejecución de Arcade en modo play: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Demostración finalizada.")
        if ui_instance is not None:
            try:
                arcade.exit()  # Intenta cerrar Arcade limpiamente
            except Exception:
                pass  # Ignorar errores al cerrar si ya está cerrado


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Entrenar o ver jugar IA de Snake (Agente v{AGENT_VERSION_TRAINER}). Datos en ./{DATA_DIR}/")
    parser.add_argument("--play", action="store_true",
                        help="Ver jugar al agente entrenado usando la UI.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT,
                        help=f"Número TOTAL de episodios objetivo para el entrenamiento (default: {NUM_EPISODES_DEFAULT}).")
    parser.add_argument("--plot", action="store_true",
                        help="Mostrar gráfica Matplotlib durante el entrenamiento.")
    parser.add_argument("--play-episodes", type=int, default=5,
                        help="Número de episodios a ver en modo play (default: 5).")
    parser.add_argument("--play-speed", type=float, default=0.05,
                        help="Velocidad de juego de la IA en la UI (segundos por paso, default: 0.05).")
    parser.add_argument("--reset", action="store_true",
                        help=f"Reiniciar entrenamiento eliminando modelo y checkpoint previos de '{DATA_DIR}/'.")

    # Hiperparámetros del Agente
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_DQN_DEFAULT,
                        help="Tasa de aprendizaje (default: %(default)s).")
    parser.add_argument("--gamma", type=float, default=DISCOUNT_FACTOR_DQN_DEFAULT,
                        help="Factor de descuento (gamma) (default: %(default)s).")
    parser.add_argument("--epsilon_start", type=float, default=EPSILON_START_DQN_DEFAULT,
                        help="Valor inicial de epsilon (default: %(default)s).")
    parser.add_argument("--epsilon_decay_rate", type=float, default=EPSILON_DECAY_DQN_DEFAULT,
                        help="Tasa de decaimiento de epsilon (default: %(default)s).")
    parser.add_argument("--epsilon_min", type=float, default=EPSILON_MIN_DQN_DEFAULT,
                        help="Valor mínimo de epsilon (default: %(default)s).")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DQN_DEFAULT,
                        help="Tamaño del batch para replay (default: %(default)s).")
    parser.add_argument("--replay_memory", type=int, default=REPLAY_MEMORY_SIZE_DQN_DEFAULT,
                        help="Capacidad de la memoria de repetición (default: %(default)s).")
    parser.add_argument("--agent_epochs", type=int, default=AGENT_EPOCHS_PER_REPLAY_DEFAULT,
                        help="Epochs por llamada a replay() (default: %(default)s).")
    parser.add_argument("--update_target_every", type=int, default=UPDATE_TARGET_EVERY_DEFAULT,
                        help="Episodios para actualizar target model (default: %(default)s).")

    # Nombre del archivo del modelo
    parser.add_argument("--model_file_base", type=str, default=MODEL_FILENAME_BASE_DEFAULT,
                        help=f"Nombre base para archivos de modelo y estado (ej: {MODEL_FILENAME_BASE_DEFAULT}). Se guardarán en '{DATA_DIR}/'.")

    args = parser.parse_args()

    current_hyperparams = {
        'lr': args.lr, 'gamma': args.gamma,
        'initial_epsilon': args.epsilon_start,
        'epsilon_decay_rate': args.epsilon_decay_rate,
        'epsilon_min': args.epsilon_min,
        'batch_size': args.batch_size,
        'replay_memory': args.replay_memory,
        'agent_epochs': args.agent_epochs,
        'update_target_every': args.update_target_every,
        # model_filename se construye dentro de run_ai_training y play_mode_with_ui usando AGENT_VERSION_TRAINER
        'model_file_base_name': args.model_file_base
    }

    if args.play:
        # Construir el path completo al modelo Keras para el modo play
        play_model_keras_filepath = os.path.join(
            DATA_DIR, f"{args.model_file_base}_{AGENT_VERSION_TRAINER}.keras")
        play_mode_with_ui(model_keras_path_full=play_model_keras_filepath,
                          num_episodes_to_play=args.play_episodes,
                          play_speed_ui=args.play_speed)
    else:
        run_ai_training(num_total_episodes_target=args.episodes,
                        reset_model_flag=args.reset,
                        hyperparams=current_hyperparams,
                        show_plot=args.plot)
