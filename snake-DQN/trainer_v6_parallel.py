# trainer_v6_parallel.py
# Basado en trainer_v5.py y la lógica de workers de trainer_v6.py.
# Usa DQNAgent de agent_v6_parallel.py (modelo Sequential).
# MODIFICADO: Workers ahora envían batches de experiencias.

# Establecer ANTES de cualquier import de TensorFlow
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
from snake_logic_v11 import SnakeLogic, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC
from agent_v6_parallel import DQNAgent, AGENT_VERSION, DATA_DIR
import random
import queue  # Para la excepción Empty en colas y queue.Full
import multiprocessing as mp
from collections import deque
import pickle
import argparse
import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Imports estándar

# Imports de la aplicación


# Variable global para gpus, se llenará en setup_tensorflow_main_process
gpus_devices_list = []

# --- Funciones de configuración para el proceso principal ---


def setup_tensorflow_main_process():
    global gpus_devices_list
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print(
                f"INFO: Política de precisión mixta global establecida: {mixed_precision.global_policy().name}")
        except Exception as e:
            print(
                f"WARN: No se pudo establecer mixed_float16: {e}. Usando float32 por defecto.")

        gpus_devices_list = tf.config.list_physical_devices('GPU')
        if gpus_devices_list:
            try:
                for gpu_dev in gpus_devices_list:
                    tf.config.experimental.set_memory_growth(gpu_dev, True)
                print(
                    f"INFO: Crecimiento de memoria GPU habilitado para {len(gpus_devices_list)} GPU(s).")
                tf.config.set_visible_devices(gpus_devices_list[0], 'GPU')
                print(
                    f"INFO: Proceso principal configurado para usar GPU: {gpus_devices_list[0].name}")
            except RuntimeError as e:
                print(
                    f"ERROR: Error configurando GPU para proceso principal: {e}")
        else:
            print(
                "INFO: No se detectaron GPUs por TensorFlow. El proceso principal se ejecutará en CPU.")
    except ImportError:
        print(
            "CRITICAL ERROR: TensorFlow no está instalado o no se pudo importar. Saliendo.")
        exit(1)


def setup_matplotlib_main_process():
    try:
        if 'DISPLAY' not in os.environ and os.name != 'nt':
            print("INFO: Configurando Matplotlib para backend 'Agg' (no interactivo).")
            mpl.use('Agg')
    except ImportError:
        print("WARN: No se pudo importar matplotlib o establecer backend 'Agg'.")


# --- Constantes y Hiperparámetros por Defecto ---
STATE_SIZE = 11
ACTION_SIZE = 4
LEARNING_RATE_DQN_DEFAULT = 0.0001
DISCOUNT_FACTOR_DQN_DEFAULT = 0.99
EPSILON_START_DQN_DEFAULT = 1.0
# Más lento para decaimiento por replay (ajustado)
EPSILON_DECAY_RATE_GLOBAL_DEFAULT = 0.99996
EPSILON_MIN_GLOBAL_DEFAULT = 0.01         # Mínimo para el learner
EPSILON_DECAY_RATE_WORKER_DEFAULT = 0.9995  # Por episodio de worker (como v5)
EPSILON_MIN_WORKER_DEFAULT = 0.01         # Mínimo para workers (ajustado)

REPLAY_MEMORY_SIZE_DQN_DEFAULT = 500_000
BATCH_SIZE_DQN_DEFAULT = 4096           # Coincidir con v5 funcional
AGENT_EPOCHS_PER_REPLAY_DEFAULT = 1
UPDATE_TARGET_AND_SYNC_WORKERS_EVERY_N_TRAIN_STEPS = 250
LEARN_EVERY_N_GLOBAL_STEPS = 4
# Cuántas experiencias agrupa un worker antes de enviar
EXPERIENCE_BATCH_FROM_WORKER_SIZE = 32

MODEL_FILENAME_BASE_DEFAULT = "dqn_snake_checkpoint"
NUM_EPISODES_DEFAULT = 50000
SAVE_CHECKPOINT_EVERY_DEFAULT = 1000
PRINT_STATS_EVERY_DEFAULT = 100
NUM_WORKERS_DEFAULT = max(1, mp.cpu_count() // 2)
if NUM_WORKERS_DEFAULT == 0 and mp.cpu_count() > 1:
    NUM_WORKERS_DEFAULT = 1

plt.ion()  # type: ignore
fig_plot, ax_plot = None, None


def setup_matplotlib_plot():
    global fig_plot, ax_plot
    # type: ignore
    if fig_plot is None or not plt.fignum_exists(fig_plot.number):
        fig_plot, ax_plot = plt.subplots(figsize=(12, 6))  # type: ignore
    else:
        ax_plot.clear()  # type: ignore
    ax_plot.set_title(
        f"Entrenamiento Snake IA (DQN v{AGENT_VERSION} - MP)")  # type: ignore
    ax_plot.set_xlabel("Episodio Global")
    ax_plot.set_ylabel("Puntuación")  # type: ignore
    line_score, = ax_plot.plot(
        [], [], 'b-', alpha=0.4, label='Puntuación Episodio (Worker)')  # type: ignore
    line_avg_score, = ax_plot.plot(
        [], [], 'r-', linewidth=2, label='Media Puntuación (últ. 100)')  # type: ignore
    ax_plot.legend()
    fig_plot.tight_layout()  # type: ignore
    return line_score, line_avg_score


def update_matplotlib_plot(plot_scores_data, line_score, line_avg_score):
    if not plot_scores_data or ax_plot is None or fig_plot is None:
        return  # type: ignore
    episodes_x = [item[0] for item in plot_scores_data]
    scores_y = [item[1] for item in plot_scores_data]
    line_score.set_data(episodes_x, scores_y)
    if len(scores_y) >= 1:
        avg_scores_plot = [np.mean(scores_y[max(0, i-99):i+1])
                           for i in range(len(scores_y))]
        line_avg_score.set_data(episodes_x, avg_scores_plot)
    ax_plot.relim()
    ax_plot.autoscale_view(True, True, True)  # type: ignore
    try:
        fig_plot.canvas.draw_idle()
        fig_plot.canvas.flush_events()  # type: ignore
    except Exception as e:
        print(f"WARN: Error actualizando matplotlib: {e}")
    plt.pause(0.0001)

# --- Worker para Multiprocessing ---


def build_worker_keras_model_sequential(state_size, action_size, learning_rate_dummy=0.001):
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential
    model = Sequential([
        Input(shape=(state_size,)),
        Dense(512, activation='relu', dtype='float32'),
        Dense(512, activation='relu', dtype='float32'),
        Dense(256, activation='relu', dtype='float32'),
        Dense(action_size, activation='linear', dtype='float32')
    ])
    optimizer = Adam(learning_rate=learning_rate_dummy)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def snake_worker_process_fn(worker_id, env_config, experience_q, model_weights_q, stop_signal,
                            epsilon_config, state_dim, action_dim, experience_batch_send_size):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    import numpy as np
    import random
    from snake_logic_v11 import SnakeLogic
    import queue

    tf.config.set_visible_devices([], 'GPU')
    env = SnakeLogic(env_config['width'],
                     env_config['height'], env_config['step_size'])
    worker_model = build_worker_keras_model_sequential(state_dim, action_dim)

    try:
        initial_weights = model_weights_q.get(timeout=15)
        if initial_weights:
            worker_model.set_weights(initial_weights)
    except queue.Empty:
        print(
            f"[Worker {worker_id}] WARN: No recibió pesos iniciales tras 15s.")
    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: Cargando pesos iniciales: {e}")

    local_epsilon = epsilon_config['initial_epsilon_worker']
    local_experience_buffer = []

    while not stop_signal.is_set():
        try:
            new_weights = model_weights_q.get_nowait()
            if new_weights:
                worker_model.set_weights(new_weights)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[Worker {worker_id}] WARN: Error obteniendo pesos: {e}")

        current_state_tuple = env.reset()
        episode_reward_sum = 0.0
        done = False
        steps_in_episode = 0
        steps_since_last_food = 0
        max_steps_this_episode = (env_config['width']//env_config['step_size'])*(
            env_config['height']//env_config['step_size'])*2.5
        max_steps_this_episode = max(max_steps_this_episode, 100)
        max_steps_no_food_limit = (
            env_config['width']+env_config['height'])//env_config['step_size']

        while not done and not stop_signal.is_set():
            if random.random() <= local_epsilon:
                action_idx = random.randrange(action_dim)
            else:
                state_np_flat = np.array(
                    current_state_tuple, dtype=np.float32).flatten()
                state_tensor = tf.convert_to_tensor(
                    state_np_flat.reshape([1, state_dim]), dtype=tf.float32)
                act_values_tensor = worker_model(
                    state_tensor, training=False)  # type: ignore
                action_idx = np.argmax(act_values_tensor[0].numpy())

            next_state_tuple, reward, done, info = env.step(action_idx)
            episode_reward_sum += reward
            current_reward_for_memory = reward
            if info.get('ate_food', False):
                steps_since_last_food = 0
            else:
                steps_since_last_food += 1
            if not info.get('ate_food', False) and steps_since_last_food > max_steps_no_food_limit:
                current_reward_for_memory -= 50
                done = True

            current_state_flat = np.array(
                current_state_tuple, dtype=np.float32).flatten()
            next_state_flat = np.array(
                next_state_tuple, dtype=np.float32).flatten()

            experience_package = (
                current_state_flat, action_idx, current_reward_for_memory,
                next_state_flat, done, env.score if done else None
            )
            local_experience_buffer.append(experience_package)

            if len(local_experience_buffer) >= experience_batch_send_size or \
               (done and len(local_experience_buffer) > 0):
                try:
                    experience_q.put(
                        list(local_experience_buffer), timeout=1.0)
                    local_experience_buffer.clear()
                except queue.Full:
                    local_experience_buffer.clear()
                except Exception as e_put_batch:
                    print(
                        f"[Worker {worker_id}] ERROR: Poniendo batch de exp en cola: {e_put_batch}.")
                    stop_signal.set()
                    break

            current_state_tuple = next_state_tuple
            steps_in_episode += 1
            if steps_in_episode >= max_steps_this_episode:
                done = True

            if done:
                if local_epsilon > epsilon_config['epsilon_min_worker']:
                    local_epsilon *= epsilon_config['epsilon_decay_rate_worker']
                    local_epsilon = max(
                        epsilon_config['epsilon_min_worker'], local_epsilon)

        if stop_signal.is_set():
            break

# --- Función Principal de Entrenamiento ---


def run_ai_training(num_total_episodes_target, reset_model_flag, hyperparams, show_plot_flag, num_workers_to_use):
    plot_scores_data_session = []
    model_base_filename = hyperparams['model_file_base_name']
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"INFO: Directorio de datos creado: {DATA_DIR}")
        except OSError as e:
            print(f"ERROR: Crítico creando dir {DATA_DIR}: {e}. Saliendo.")
            return

    model_keras_file = os.path.join(
        DATA_DIR, f"{model_base_filename}_{AGENT_VERSION}.keras")
    training_state_file = os.path.join(
        DATA_DIR, f"{model_base_filename}_{AGENT_VERSION}_train_state.pkl")

    current_global_episode = 1
    current_global_step = 0
    num_training_steps_done = 0

    if reset_model_flag:
        print(f"INFO: --reset activado. Eliminando checkpoints...")
        if os.path.exists(model_keras_file):
            os.remove(model_keras_file)
        if os.path.exists(training_state_file):
            os.remove(training_state_file)

    main_learner_agent = DQNAgent(
        state_size=STATE_SIZE, action_size=ACTION_SIZE,
        learning_rate=hyperparams['lr'], discount_factor=hyperparams['gamma'],
        epsilon=hyperparams['initial_epsilon_global'],
        epsilon_decay_rate=hyperparams['epsilon_decay_rate_global'],
        epsilon_min=hyperparams['epsilon_min_global'],
        replay_memory_size=hyperparams['replay_memory'],
        batch_size=hyperparams['batch_size'],
        model_filepath=model_keras_file,
        epochs_per_replay=hyperparams['agent_epochs']
    )

    initial_global_step_for_session = 0
    initial_train_steps_for_session = 0

    if not reset_model_flag and os.path.exists(training_state_file):
        print(
            f"INFO: Intentando cargar estado de entrenamiento: {training_state_file}...")
        try:
            with open(training_state_file, 'rb') as f:
                training_state_loaded = pickle.load(f)
            current_global_episode = training_state_loaded.get(
                'last_completed_episode_global', 0) + 1
            plot_scores_data_session = training_state_loaded.get(
                'plot_scores_history', [])
            main_learner_agent.epsilon = training_state_loaded.get(
                'agent_epsilon_global', main_learner_agent.initial_epsilon)
            loaded_memory_data = training_state_loaded.get('agent_memory', [])
            main_learner_agent.memory = deque(
                loaded_memory_data, maxlen=main_learner_agent.replay_memory_capacity)
            current_global_step = training_state_loaded.get(
                'global_step_counter', 0)
            num_training_steps_done = training_state_loaded.get(
                'training_steps_done', 0)
            initial_global_step_for_session = current_global_step  # Para stats de sesión
            initial_train_steps_for_session = num_training_steps_done  # Para stats de sesión
            print(
                f"INFO: Estado de entrenamiento cargado. Continuando desde episodio global {current_global_episode}.")
            print(
                f"  Epsilon global del learner: {main_learner_agent.epsilon:.4f}, Tamaño de memoria: {len(main_learner_agent.memory)}")
        except Exception as e:
            print(
                f"WARN: Error al cargar estado de entrenamiento: {e}. Iniciando desde cero.")
            current_global_episode = 1
            plot_scores_data_session = []
            current_global_step = 0
            num_training_steps_done = 0
            main_learner_agent.reset_epsilon_and_memory(
                hyperparams['initial_epsilon_global'])
    elif not reset_model_flag:
        print(
            f"INFO: No se encontró {training_state_file}. Iniciando nuevo entrenamiento desde episodio 1.")
    else:
        print("INFO: Iniciando entrenamiento desde episodio 1 debido a --reset.")
        main_learner_agent.reset_epsilon_and_memory(
            hyperparams['initial_epsilon_global'])

    experience_submission_q = mp.Queue(
        maxsize=num_workers_to_use * 10 * hyperparams['experience_batch_send_size'])  # Ajustar maxsize
    model_weights_broadcast_qs = [
        mp.Queue(maxsize=1) for _ in range(num_workers_to_use)]
    stop_workers_event = mp.Event()
    env_runtime_params = {'width': SCREEN_WIDTH_LOGIC,
                          'height': SCREEN_HEIGHT_LOGIC, 'step_size': STEP_SIZE_LOGIC}
    worker_eps_params = {
        'initial_epsilon_worker': hyperparams['initial_epsilon_worker'],
        'epsilon_decay_rate_worker': hyperparams['epsilon_decay_rate_worker'],
        'epsilon_min_worker': hyperparams['epsilon_min_worker']
    }
    active_workers_list = []
    for i in range(num_workers_to_use):
        p = mp.Process(target=snake_worker_process_fn, args=(
            i, env_runtime_params, experience_submission_q, model_weights_broadcast_qs[
                i], stop_workers_event,
            # Pasar tamaño de batch
            worker_eps_params, STATE_SIZE, ACTION_SIZE, hyperparams['experience_batch_send_size']
        ))
        active_workers_list.append(p)
        p.start()
    print(f"INFO: {num_workers_to_use} procesos worker iniciados.")

    time.sleep(5)
    if hasattr(main_learner_agent.model, 'get_weights'):
        initial_weights_to_send = main_learner_agent.model.get_weights()
        for i in range(num_workers_to_use):
            try:
                model_weights_broadcast_qs[i].put(
                    initial_weights_to_send, timeout=2)
            except queue.Full:
                print(f"WARN: Cola de pesos worker {i} llena (inicial).")
            except Exception as e:
                print(f"ERROR: Enviando pesos iniciales worker {i}: {e}")
    else:
        print("CRITICAL ERROR: main_learner_agent.model no tiene 'get_weights'. Saliendo.")
        stop_workers_event.set()
        for p_w in active_workers_list:
            p_w.join(timeout=1)
            return

    recent_scores_for_avg = deque(maxlen=100)
    if plot_scores_data_session:
        recent_scores_for_avg.extend(
            [s[1] for s in plot_scores_data_session if len(s) == 2])
    live_plot_line_score, live_plot_line_avg_score = None, None  # type: ignore
    if show_plot_flag:
        live_plot_line_score, live_plot_line_avg_score = setup_matplotlib_plot()
        if plot_scores_data_session:
            update_matplotlib_plot(
                plot_scores_data_session, live_plot_line_score, live_plot_line_avg_score)

    print(
        f"\nINFO: Iniciando/Continuando entrenamiento {AGENT_VERSION} hasta {num_total_episodes_target} ep globales.")
    try:
        import tensorflow as tf
        current_lr_val = tf.keras.backend.get_value(
            main_learner_agent.model.optimizer.learning_rate)
    except:
        current_lr_val = main_learner_agent.learning_rate
    print(
        f"INFO: Hiperparámetros: LR={current_lr_val:.6f}, Gamma={main_learner_agent.gamma}, EpsL:{main_learner_agent.epsilon:.3f}, BatchS:{main_learner_agent.batch_size}")
    print(
        f"INFO: Target/Sync Freq (Train Steps):{hyperparams['update_target_every_n_steps']}, Learn Freq (Global Env Steps):{hyperparams['learn_every_n_steps']}")

    session_start_time = time.time()
    run_ai_training.last_print_time_attr = session_start_time  # type: ignore
    run_ai_training.last_global_step_print = current_global_step  # type: ignore
    training_manually_halted = False

    try:
        while current_global_episode <= num_total_episodes_target:
            try:
                experience_batch_from_worker = experience_submission_q.get(
                    timeout=0.1)

                for exp_package_tuple in experience_batch_from_worker:
                    exp_state, exp_action, exp_reward, exp_next_state, exp_done, exp_score_if_done = exp_package_tuple
                    main_learner_agent.remember(
                        exp_state, exp_action, exp_reward, exp_next_state, exp_done)
                    current_global_step += 1

                    if exp_done:
                        plot_scores_data_session.append(
                            (current_global_episode, exp_score_if_done))
                        recent_scores_for_avg.append(exp_score_if_done)

                        if current_global_episode % PRINT_STATS_EVERY_DEFAULT == 0:
                            current_time_print = time.time()
                            elapsed_time_interval = current_time_print - \
                                run_ai_training.last_print_time_attr  # type: ignore
                            run_ai_training.last_print_time_attr = current_time_print  # type: ignore
                            steps_this_interval = current_global_step - \
                                run_ai_training.last_global_step_print  # type: ignore
                            run_ai_training.last_global_step_print = current_global_step  # type: ignore
                            sps_val = steps_this_interval / \
                                elapsed_time_interval if elapsed_time_interval > 0.001 else 0.0
                            avg_score_val = np.mean(
                                recent_scores_for_avg) if recent_scores_for_avg else 0.0
                            print(f"EpG:{current_global_episode}/{num_total_episodes_target} StG:{current_global_step}({sps_val:.0f}SPS) "
                                  f"S:{exp_score_if_done} EpsL:{main_learner_agent.epsilon:.3f} "
                                  f"AvgS(100):{avg_score_val:.2f} Mem:{len(main_learner_agent.memory)//1000}k TrainS:{num_training_steps_done//1000}k")
                            if show_plot_flag and live_plot_line_score is not None:
                                update_matplotlib_plot(
                                    plot_scores_data_session, live_plot_line_score, live_plot_line_avg_score)

                        current_global_episode += 1

                        if current_global_episode % SAVE_CHECKPOINT_EVERY_DEFAULT == 0 or \
                           current_global_episode > num_total_episodes_target:
                            print(
                                f"\nINFO: Guardando checkpoint en episodio global {current_global_episode-1}...")
                            main_learner_agent.save_keras_model()
                            training_state_to_save = {
                                'last_completed_episode_global': current_global_episode - 1,
                                'agent_epsilon_global': main_learner_agent.epsilon,
                                'agent_memory': list(main_learner_agent.memory),
                                'plot_scores_history': plot_scores_data_session,
                                'global_step_counter': current_global_step,
                                'training_steps_done': num_training_steps_done,
                            }
                            with open(training_state_file, 'wb') as f:
                                pickle.dump(training_state_to_save, f)
                            print(f"INFO: Checkpoint guardado.\n")

                if len(main_learner_agent.memory) >= main_learner_agent.batch_size and \
                   current_global_step % hyperparams['learn_every_n_steps'] == 0:
                    main_learner_agent.replay()
                    num_training_steps_done += 1
                    if main_learner_agent.epsilon > hyperparams['epsilon_min_global']:
                        main_learner_agent.epsilon *= hyperparams['epsilon_decay_rate_global']
                        main_learner_agent.epsilon = max(
                            hyperparams['epsilon_min_global'], main_learner_agent.epsilon)

                    if num_training_steps_done > 0 and \
                       num_training_steps_done % hyperparams['update_target_every_n_steps'] == 0:
                        main_learner_agent.update_target_model()
                        print(
                            f"INFO: Target actualizado y pesos enviados (TS:{num_training_steps_done}). Learner Eps:{main_learner_agent.epsilon:.3f}")
                        updated_weights_for_workers = main_learner_agent.model.get_weights()
                        for i in range(num_workers_to_use):
                            try:
                                while not model_weights_broadcast_qs[i].empty():
                                    model_weights_broadcast_qs[i].get_nowait()
                                model_weights_broadcast_qs[i].put(
                                    updated_weights_for_workers, block=False)
                            except queue.Full:
                                pass
                            except Exception as e_q_put:
                                print(
                                    f"WARN: Enviando pesos worker {i}: {e_q_put}")

            except queue.Empty:
                time.sleep(0.001)
                pass
            except KeyboardInterrupt:
                print("\nINFO: Entrenamiento interrumpido.")
                training_manually_halted = True
                break
            except Exception as e_main_loop:
                print(f"ERROR: Inesperado: {e_main_loop}")
                import traceback
                traceback.print_exc()
                training_manually_halted = True
                break

        if current_global_episode > num_total_episodes_target:
            print("\nINFO: Target episodios alcanzado.")
    finally:
        print("\nINFO: Finalizando sesión de entrenamiento...")
        stop_workers_event.set()
        for i, p_worker_proc in enumerate(active_workers_list):
            p_worker_proc.join(timeout=5)
            if p_worker_proc.is_alive():
                print(f"WARN: Worker {i} no terminó, forzando...")
                p_worker_proc.terminate()
                p_worker_proc.join()
        print(
            f"INFO: Todos los {len(active_workers_list)} workers finalizados.")
        if hasattr(experience_submission_q, 'close'):
            experience_submission_q.close()
        if hasattr(experience_submission_q, 'join_thread'):
            experience_submission_q.join_thread()
        for q_w_bc in model_weights_broadcast_qs:
            if hasattr(q_w_bc, 'close'):
                q_w_bc.close()
            if hasattr(q_w_bc, 'join_thread'):
                q_w_bc.join_thread()
        session_end_time = time.time()
        total_duration_minutes = (session_end_time - session_start_time) / 60
        print(
            f"INFO: Sesión {'detenida manualmente' if training_manually_halted else 'finalizada'}.")
        print(
            f"INFO: Episodios globales completados: {current_global_episode -1}")
        print(
            f"INFO: Duración total de ESTA SESIÓN: {total_duration_minutes:.2f} minutos.")

        steps_this_session = current_global_step - initial_global_step_for_session
        train_steps_this_session = num_training_steps_done - initial_train_steps_for_session
        print(
            f"INFO: Pasos globales de entorno procesados en esta sesión: {steps_this_session}")
        print(
            f"INFO: Pasos de entrenamiento (replay) realizados en esta sesión: {train_steps_this_session}")

        final_save_episode_num = current_global_episode - 1
        if hasattr(main_learner_agent, 'model') and main_learner_agent.model and final_save_episode_num >= 0:
            print(
                f"INFO: Guardando estado final en episodio global {final_save_episode_num}...")
            main_learner_agent.save_keras_model()
            final_training_state_to_save = {
                'last_completed_episode_global': final_save_episode_num,
                'agent_epsilon_global': main_learner_agent.epsilon,
                'agent_memory': list(main_learner_agent.memory),
                'plot_scores_history': plot_scores_data_session,
                'global_step_counter': current_global_step,
                'training_steps_done': num_training_steps_done,
            }
            try:
                with open(training_state_file, 'wb') as f:
                    pickle.dump(final_training_state_to_save, f)
                print(
                    f"INFO: Estado final del entrenamiento guardado en {training_state_file}.")
            except Exception as e_final_save:
                print(f"ERROR: Al guardar el estado final: {e_final_save}")
        if show_plot_flag and fig_plot is not None:  # type: ignore
            plt.ioff()  # type: ignore
            try:
                if plt.fignum_exists(fig_plot.number):  # type: ignore
                    plot_save_filename = f"dqn_training_plot_{model_base_filename}_{AGENT_VERSION}_ep{final_save_episode_num}.png"
                    plot_save_path = os.path.join(DATA_DIR, plot_save_filename)
                    fig_plot.savefig(plot_save_path)
                    print(f"INFO: Gráfica guardada: {plot_save_path}")
                    plt.close(fig_plot)  # type: ignore
            except Exception as e_plot_save:
                print(f"WARN: Error guardando/cerrando gráfica: {e_plot_save}")
        print("INFO: Proceso principal de entrenamiento finalizado.")


# --- Modo Play (Adaptado para agent_v6_mp) ---
def play_mode_with_ui_v6_mp(model_keras_path_full, num_episodes_to_play, play_speed_delay):
    try:
        import arcade  # type: ignore
        from snake_ui_v10 import SnakeGameUI
    except ImportError as e_arcade:
        print(
            f"ERROR: Al importar Arcade o SnakeGameUI: {e_arcade}. Saliendo del modo play.")
        return

    print(
        f"\n--- Viendo jugar al agente DQN v{AGENT_VERSION} ({model_keras_path_full}) con UI ---")
    if not os.path.exists(model_keras_path_full):
        print(
            f"ERROR: No se encontró el modelo Keras en '{model_keras_path_full}'. Saliendo.")
        return

    game_logic_instance = SnakeLogic(
        SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    agent_for_play = DQNAgent(
        state_size=STATE_SIZE, action_size=ACTION_SIZE,
        epsilon=0.01, epsilon_min=0.01, model_filepath=model_keras_path_full, learning_rate=0.0001
    )
    if not agent_for_play.model:
        print(
            f"ERROR: No se pudo cargar el modelo del agente desde {model_keras_path_full}. Saliendo.")
        return

    ui_instance_obj = None
    try:
        print(f"INFO: Iniciando UI para demostración.")
        ui_instance_obj = SnakeGameUI(
            game_logic_instance, agent_instance=agent_for_play,
            num_demo_episodes=num_episodes_to_play, play_speed=play_speed_delay, ai_controlled_demo=True
        )
        arcade.run()
    except Exception as e_arcade_run:
        print(f"ERROR: Durante Arcade en modo play: {e_arcade_run}")
        import traceback
        traceback.print_exc()
    finally:
        print("INFO: Demostración finalizada.")
        if ui_instance_obj:
            try:
                arcade.exit()
            except Exception:
                pass


# --- Parseo de Argumentos y Ejecución ---
if __name__ == "__main__":
    try:
        current_mp_context = mp.get_context()
        if current_mp_context.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
            print("INFO: Método de inicio de multiprocessing establecido en 'spawn'.")
        elif current_mp_context.get_start_method() != 'spawn':
            try:
                mp.set_start_method('spawn', force=True)
                print("INFO: Método de inicio de multiprocessing forzado a 'spawn'.")
            except RuntimeError as e_mp_force_spawn:
                print(
                    f"WARN: No se pudo forzar mp a 'spawn': {e_mp_force_spawn}. Usando: {current_mp_context.get_start_method()}")
        else:
            print(
                f"INFO: Método de inicio de multiprocessing ya es '{current_mp_context.get_start_method()}'.")
    except Exception as e_set_mp_method:
        print(f"WARN: Excepción estableciendo mp: {e_set_mp_method}")

    mp.freeze_support()
    setup_tensorflow_main_process()
    setup_matplotlib_main_process()

    parser = argparse.ArgumentParser(
        description=f"Entrenar o ver IA Snake (v{AGENT_VERSION} - DQN Seq + MP)")
    parser.add_argument("--play", action="store_true", help="Ver jugar.")
    parser.add_argument("--reset", action="store_true",
                        help="Reiniciar entrenamiento.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS_DEFAULT)
    parser.add_argument("--plot", action="store_true", help="Mostrar gráfica.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_DQN_DEFAULT)
    parser.add_argument("--gamma", type=float,
                        default=DISCOUNT_FACTOR_DQN_DEFAULT)
    parser.add_argument("--epsilon_start", type=float,
                        default=EPSILON_START_DQN_DEFAULT)
    parser.add_argument("--epsilon_decay_global", type=float,
                        default=EPSILON_DECAY_RATE_GLOBAL_DEFAULT)
    parser.add_argument("--epsilon_min_global", type=float,
                        default=EPSILON_MIN_GLOBAL_DEFAULT)
    parser.add_argument("--epsilon_decay_worker", type=float,
                        default=EPSILON_DECAY_RATE_WORKER_DEFAULT)
    parser.add_argument("--epsilon_min_worker", type=float,
                        default=EPSILON_MIN_WORKER_DEFAULT)
    parser.add_argument("--batch_size", type=int,
                        default=BATCH_SIZE_DQN_DEFAULT)
    parser.add_argument("--replay_memory", type=int,
                        default=REPLAY_MEMORY_SIZE_DQN_DEFAULT)
    parser.add_argument("--agent_epochs", type=int,
                        default=AGENT_EPOCHS_PER_REPLAY_DEFAULT)
    parser.add_argument("--update_target_train_steps", type=int,
                        default=UPDATE_TARGET_AND_SYNC_WORKERS_EVERY_N_TRAIN_STEPS)
    parser.add_argument("--learn_global_steps", type=int,
                        default=LEARN_EVERY_N_GLOBAL_STEPS)
    parser.add_argument("--exp_batch_send_size", type=int, default=EXPERIENCE_BATCH_FROM_WORKER_SIZE,
                        help="Tamaño del lote de experiencias que envía cada worker.")
    parser.add_argument("--model_file_base", type=str,
                        default=MODEL_FILENAME_BASE_DEFAULT)
    parser.add_argument("--play-episodes", type=int, default=5)
    parser.add_argument("--play-speed", type=float, default=0.05)
    args = parser.parse_args()

    if not args.play and args.num_workers <= 0:
        print(
            f"WARN: num_workers ({args.num_workers}) inválido para entrenamiento. Usando 1 worker.")
        args.num_workers = 1
    elif args.num_workers > mp.cpu_count() * 2 and not args.play:
        print(
            f"WARN: num_workers ({args.num_workers}) muy alto (CPUs: {mp.cpu_count()}).")

    current_hyperparams_dict = {
        'lr': args.lr, 'gamma': args.gamma,
        'initial_epsilon_global': args.epsilon_start,
        'epsilon_decay_rate_global': args.epsilon_decay_global,
        'epsilon_min_global': args.epsilon_min_global,
        # Workers parten del mismo global start
        'initial_epsilon_worker': args.epsilon_start,
        'epsilon_decay_rate_worker': args.epsilon_decay_worker,
        'epsilon_min_worker': args.epsilon_min_worker,
        'batch_size': args.batch_size,
        'replay_memory': args.replay_memory,
        'agent_epochs': args.agent_epochs,
        'update_target_every_n_steps': args.update_target_train_steps,
        'learn_every_n_steps': args.learn_global_steps,
        'model_file_base_name': args.model_file_base,
        'experience_batch_send_size': args.exp_batch_send_size  # Nuevo hiperparámetro
    }

    if args.play:
        model_to_play_path = os.path.join(
            DATA_DIR, f"{args.model_file_base}_{AGENT_VERSION}.keras")
        play_mode_with_ui_v6_mp(
            model_keras_path_full=model_to_play_path,
            num_episodes_to_play=args.play_episodes,
            play_speed_delay=args.play_speed
        )
    else:
        if not hasattr(run_ai_training, 'last_global_step_print_initial'):
            run_ai_training.last_global_step_print_initial = 0  # type: ignore
        if not hasattr(run_ai_training, 'last_train_steps_print_initial'):
            run_ai_training.last_train_steps_print_initial = 0  # type: ignore
        run_ai_training(
            num_total_episodes_target=args.episodes, reset_model_flag=args.reset,
            hyperparams=current_hyperparams_dict, show_plot_flag=args.plot,
            num_workers_to_use=args.num_workers
        )
