# trainer_v6_parallel.py
# Basado en trainer_v5_parallel.py, adaptado para agent_v6_parallel y snake_logic_v12.
# Usa DQNAgent de agent_v6_parallel.py (modelo Sequential, state_size=12).
# Workers envían batches de experiencias.

import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
# Utiliza agent_v6_parallel y snake_logic_v12
from agent_v6_parallel import DQNAgent, AGENT_VERSION as AGENT_VERSION_FROM_AGENT, DATA_DIR
from snake_logic_v12 import SnakeLogic, SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC

import random
import queue
import multiprocessing as mp
from collections import deque
import pickle
import argparse
import time
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow logs set to error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


gpus_devices_list = []


def setup_tensorflow_main_process():
    global gpus_devices_list
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        try:
            # policy = mixed_precision.Policy('mixed_float16') # Optional: Enable mixed precision
            # mixed_precision.set_global_policy(policy)
            # print(f"INFO: Política de precisión mixta global establecida: {mixed_precision.global_policy().name}")
            pass  # Not enabling by default for broader compatibility
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
                # Optionally, make only one GPU visible to the main learner process
                # tf.config.set_visible_devices(gpus_devices_list[0], 'GPU')
                # print(f"INFO: Proceso principal configurado para usar GPU: {gpus_devices_list[0].name}")
            except RuntimeError as e:
                print(
                    f"ERROR: Error configurando GPU para proceso principal: {e}")
        else:
            print(
                "INFO: No se detectaron GPUs por TensorFlow. El proceso principal se ejecutará en CPU.")
    except ImportError:
        print(
            "CRITICAL ERROR: TensorFlow no está instalado. Saliendo.")
        exit(1)


def setup_matplotlib_main_process():
    try:
        if 'DISPLAY' not in os.environ and os.name != 'nt':
            print("INFO: Configurando Matplotlib para backend 'Agg' (no interactivo).")
            mpl.use('Agg')
    except ImportError:
        print("WARN: No se pudo importar matplotlib o establecer backend 'Agg'.")


# --- Constantes y Hiperparámetros ---
AGENT_VERSION_TRAINER = "6_parallel"
if AGENT_VERSION_FROM_AGENT != AGENT_VERSION_TRAINER:
    print(
        f"Advertencia: AGENT_VERSION en trainer ({AGENT_VERSION_TRAINER}) y agente ({AGENT_VERSION_FROM_AGENT}) no coinciden.")


STATE_SIZE = 12  # Clave para v6
ACTION_SIZE = 4

# Hiperparámetros (combinando trainer_v6 y trainer_v5_parallel)
LEARNING_RATE_DQN_DEFAULT = 0.0005  # De trainer_v6.py
DISCOUNT_FACTOR_DQN_DEFAULT = 0.95    # De trainer_v6.py
EPSILON_START_DQN_DEFAULT = 1.0     # Común

# Epsilon global para el learner (más lento, decae por paso de entrenamiento)
EPSILON_DECAY_RATE_GLOBAL_DEFAULT = 0.99996  # De trainer_v5_parallel.py
EPSILON_MIN_GLOBAL_DEFAULT = 0.01          # Común

# Epsilon para workers (más rápido, decae por episodio de worker)
EPSILON_DECAY_RATE_WORKER_DEFAULT = 0.9995  # De trainer_v6.py (decay original)
# Workers pueden explorar más, o 0.01 como antes
EPSILON_MIN_WORKER_DEFAULT = 0.1

# De trainer_v5_parallel.py (menor que v6 single)
REPLAY_MEMORY_SIZE_DQN_DEFAULT = 500_000
BATCH_SIZE_DQN_DEFAULT = 64            # Común
AGENT_EPOCHS_PER_REPLAY_DEFAULT = 1      # Común

UPDATE_TARGET_AND_SYNC_WORKERS_EVERY_N_TRAIN_STEPS = 250  # De trainer_v5_parallel.py
# De trainer_v5_parallel.py
LEARN_EVERY_N_GLOBAL_STEPS = 4
EXPERIENCE_BATCH_FROM_WORKER_SIZE = 64                 # Aumentado de 32

MODEL_FILENAME_BASE_DEFAULT = "dqn_snake_checkpoint"
NUM_EPISODES_DEFAULT = 100000  # De trainer_v6.py
SAVE_CHECKPOINT_EVERY_DEFAULT = 1000  # Era 500 en v6, 1000 en v5_parallel
# Episodios globales, era 10 en v6, 100 en v5_parallel
PRINT_STATS_EVERY_DEFAULT = 100

# Ajustado para no saturar CPUs pequeñas
NUM_WORKERS_DEFAULT = max(1, mp.cpu_count() // 2 - 1)
if NUM_WORKERS_DEFAULT == 0 and mp.cpu_count() > 1:
    NUM_WORKERS_DEFAULT = 1
if NUM_WORKERS_DEFAULT == 0 and mp.cpu_count() == 1:
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
        f"Entrenamiento Snake IA (DQN v{AGENT_VERSION_TRAINER} - MP)")  # type: ignore
    ax_plot.set_xlabel("Episodio Global")  # type: ignore
    ax_plot.set_ylabel("Puntuación")  # type: ignore
    line_score, = ax_plot.plot(  # type: ignore
        [], [], marker='.', linestyle='-', color='b', alpha=0.3, markersize=2, label='Puntuación Episodio (Worker)')
    line_avg_score, = ax_plot.plot(  # type: ignore
        [], [], 'r-', linewidth=2, label='Media Puntuación (últ. 100)')
    ax_plot.legend()  # type: ignore
    fig_plot.tight_layout()  # type: ignore
    return line_score, line_avg_score


def update_matplotlib_plot(plot_scores_data, line_score, line_avg_score):
    if not plot_scores_data or ax_plot is None or fig_plot is None:  # type: ignore
        return
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
        print(f"WARN: Error actualizando matplotlib: {e}")
    plt.pause(0.0001)


def build_worker_keras_model(state_size, action_size, learning_rate_dummy=0.001):
    # Esta función crea un modelo con la misma arquitectura que _build_model en DQNAgent
    # pero es usada por los workers que no necesitan la clase DQNAgent completa.
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(state_size,)),  # state_size será 12 para v6
        Dense(512, activation='relu', dtype='float32'),
        Dense(512, activation='relu', dtype='float32'),
        Dense(256, activation='relu', dtype='float32'),
        Dense(action_size, activation='linear', dtype='float32')
    ])
    # El LR no importa mucho aquí, los pesos se sincronizan
    optimizer = Adam(learning_rate=learning_rate_dummy)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def snake_worker_process_fn(worker_id, env_config, experience_q, model_weights_q, stop_signal,
                            epsilon_config, state_dim, action_dim, experience_batch_send_size):
    # Configuración específica del worker
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    import numpy as np
    import random
    # Importar SnakeLogic dentro del worker para evitar problemas de serialización
    from snake_logic_v12 import SnakeLogic
    import queue as worker_queue  # Renombrar para evitar confusión con la variable global

    # Forzar ejecución en CPU para workers
    tf.config.set_visible_devices([], 'GPU')

    env = SnakeLogic(env_config['width'],
                     env_config['height'], env_config['step_size'])

    # Usar la función de construcción de modelo para el worker
    worker_model = build_worker_keras_model(state_dim, action_dim)

    try:
        initial_weights = model_weights_q.get(timeout=20)  # Aumentar timeout
        if initial_weights:
            worker_model.set_weights(initial_weights)
    except worker_queue.Empty:
        print(
            f"[Worker {worker_id}] WARN: No recibió pesos iniciales tras 20s.")
    except Exception as e:
        print(f"[Worker {worker_id}] ERROR: Cargando pesos iniciales: {e}")

    local_epsilon = epsilon_config['initial_epsilon_worker']
    local_experience_buffer = []

    while not stop_signal.is_set():
        try:
            # Intentar obtener nuevos pesos de forma no bloqueante
            new_weights = model_weights_q.get_nowait()
            if new_weights:
                worker_model.set_weights(new_weights)
        except worker_queue.Empty:
            pass  # Normal si no hay nuevos pesos
        except Exception as e:
            print(f"[Worker {worker_id}] WARN: Error obteniendo pesos: {e}")

        current_state_tuple = env.reset()  # state_size es 12
        # episode_reward_sum = 0.0 # No es necesario que el worker lo sume
        done = False
        # steps_in_episode = 0 # No es estrictamente necesario para el worker

        # Lógica de pasos máximos y sin comida, adaptada de trainer_v5.py / trainer_v6.py
        max_steps_this_episode = (env_config['width'] // env_config['step_size']) * \
                                 (env_config['height'] //
                                  env_config['step_size']) * 2.5
        max_steps_this_episode = int(
            max(max_steps_this_episode, 150))  # Mínimo de trainer_v6

        # Loop de un episodio para el worker
        for step_count_in_worker_episode in range(max_steps_this_episode):
            if stop_signal.is_set():
                break

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
            # episode_reward_sum += reward # No es necesario para el worker

            # El worker no necesita la lógica compleja de reward de trainer_v5 (no food limit)
            # Simplemente envía la experiencia. El learner puede tener esa lógica si se desea,
            # pero es más simple si el worker solo envía lo básico.
            # Para v6, el reward ya está definido por snake_logic_v12

            current_state_flat = np.array(
                current_state_tuple, dtype=np.float32).flatten()
            next_state_flat = np.array(
                next_state_tuple, dtype=np.float32).flatten()

            # El paquete de experiencia incluye el score si el episodio terminó
            experience_package = (
                current_state_flat, action_idx, reward,
                next_state_flat, done, env.score if done else None
            )
            local_experience_buffer.append(experience_package)

            # Enviar batch de experiencias
            if len(local_experience_buffer) >= experience_batch_send_size or \
               (done and len(local_experience_buffer) > 0):
                try:
                    experience_q.put(
                        list(local_experience_buffer), timeout=1.0)  # Enviar copia
                    local_experience_buffer.clear()
                except worker_queue.Full:
                    # print(f"[Worker {worker_id}] WARN: Cola de experiencia llena. Descartando buffer local.")
                    local_experience_buffer.clear()  # Descartar para evitar bloqueo prolongado
                except Exception as e_put_batch:
                    print(
                        f"[Worker {worker_id}] ERROR: Poniendo batch de exp en cola: {e_put_batch}.")
                    stop_signal.set()  # Detener worker en caso de error grave de cola
                    break

            current_state_tuple = next_state_tuple
            # steps_in_episode += 1

            if done:
                break  # Salir del loop de pasos del episodio

        # Decaimiento de epsilon del worker al final de su episodio
        if local_epsilon > epsilon_config['epsilon_min_worker']:
            local_epsilon *= epsilon_config['epsilon_decay_rate_worker']
            local_epsilon = max(
                epsilon_config['epsilon_min_worker'], local_epsilon)

        if stop_signal.is_set():
            break  # Salir del loop principal del worker
    print(f"[Worker {worker_id}] Terminando.")


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
        DATA_DIR, f"{model_base_filename}_{AGENT_VERSION_TRAINER}.keras")
    training_state_file = os.path.join(
        DATA_DIR, f"{model_base_filename}_{AGENT_VERSION_TRAINER}_train_state.pkl")

    current_global_episode = 1
    current_global_step = 0      # Pasos de entorno globales procesados
    num_training_steps_done = 0  # Pasos de replay/fit realizados por el learner

    if reset_model_flag:
        print(
            f"INFO: --reset activado. Eliminando checkpoints para v{AGENT_VERSION_TRAINER}...")
        if os.path.exists(model_keras_file):
            os.remove(model_keras_file)
        if os.path.exists(training_state_file):
            os.remove(training_state_file)

    # Instanciar el agente principal (learner)
    main_learner_agent = DQNAgent(
        state_size=STATE_SIZE, action_size=ACTION_SIZE,  # STATE_SIZE es 12
        learning_rate=hyperparams['lr'], discount_factor=hyperparams['gamma'],
        # Epsilon para el learner
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

            initial_global_step_for_session = current_global_step
            initial_train_steps_for_session = num_training_steps_done
            print(
                f"INFO: Estado de entrenamiento cargado. Continuando desde episodio global {current_global_episode}.")
            print(
                f"  Epsilon global del learner: {main_learner_agent.epsilon:.4f}, Memoria: {len(main_learner_agent.memory)}")
        except Exception as e:
            print(
                f"WARN: Error al cargar estado: {e}. Iniciando desde cero.")
            current_global_episode = 1
            plot_scores_data_session = []
            current_global_step = 0
            num_training_steps_done = 0
            main_learner_agent.reset_epsilon_and_memory(
                hyperparams['initial_epsilon_global'])
    elif not reset_model_flag:
        print(
            f"INFO: No se encontró {training_state_file}. Iniciando nuevo entrenamiento.")
        main_learner_agent.reset_epsilon_and_memory(
            hyperparams['initial_epsilon_global'])
    else:  # reset_model_flag is True
        print("INFO: Iniciando entrenamiento desde episodio 1 debido a --reset.")
        main_learner_agent.reset_epsilon_and_memory(
            hyperparams['initial_epsilon_global'])

    # Configuración de Multiprocessing
    # Ajustar maxsize de cola de experiencia
    experience_submission_q = mp.Queue(
        maxsize=num_workers_to_use * 2 * hyperparams['experience_batch_send_size'])
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
            i, env_runtime_params, experience_submission_q, model_weights_broadcast_qs[i],
            stop_workers_event, worker_eps_params,
            STATE_SIZE, ACTION_SIZE, hyperparams['experience_batch_send_size']
        ), daemon=True)  # Daemon para que terminen con el principal
        active_workers_list.append(p)
        p.start()
    print(f"INFO: {num_workers_to_use} procesos worker iniciados.")

    time.sleep(5)  # Dar tiempo a los workers para inicializar
    if hasattr(main_learner_agent.model, 'get_weights'):
        initial_weights_to_send = main_learner_agent.model.get_weights()
        for i in range(num_workers_to_use):
            try:
                # Limpiar cola por si acaso antes de poner nuevos pesos
                while not model_weights_broadcast_qs[i].empty():
                    model_weights_broadcast_qs[i].get_nowait()
                model_weights_broadcast_qs[i].put(
                    initial_weights_to_send, timeout=5)
            except queue.Full:
                print(
                    f"WARN: Cola de pesos worker {i} llena al enviar pesos iniciales.")
            except Exception as e:
                print(f"ERROR: Enviando pesos iniciales a worker {i}: {e}")
    else:
        print("CRITICAL ERROR: main_learner_agent.model no tiene 'get_weights'. Saliendo.")
        stop_workers_event.set()
        for p_w in active_workers_list:
            p_w.join(timeout=1)
        return

    recent_scores_for_avg = deque(maxlen=100)
    if plot_scores_data_session:  # Poblar con datos cargados
        recent_scores_for_avg.extend(
            [s[1] for s in plot_scores_data_session if len(s) == 2])

    live_plot_line_score, live_plot_line_avg_score = None, None  # type: ignore
    if show_plot_flag:
        live_plot_line_score, live_plot_line_avg_score = setup_matplotlib_plot()
        if plot_scores_data_session:
            update_matplotlib_plot(
                plot_scores_data_session, live_plot_line_score, live_plot_line_avg_score)

    print(
        f"\nINFO: Iniciando/Continuando entrenamiento {AGENT_VERSION_TRAINER} hasta {num_total_episodes_target} ep globales.")
    try:
        import tensorflow as tf  # Para obtener LR del optimizador
        current_lr_val = tf.keras.backend.get_value(
            main_learner_agent.model.optimizer.learning_rate)
    except:
        current_lr_val = main_learner_agent.learning_rate  # Fallback

    print(f"INFO: Hiperparams: LR={current_lr_val:.6f}, Gamma={main_learner_agent.gamma:.3f}, EpsL:{main_learner_agent.epsilon:.3f} (MinG: {hyperparams['epsilon_min_global']:.3f}), "
          f"BatchS:{main_learner_agent.batch_size}, MemCap:{main_learner_agent.memory.maxlen//1000}k")
    print(
        f"INFO: WorkerEps (Init/Decay/Min): {hyperparams['initial_epsilon_worker']:.2f}/{hyperparams['epsilon_decay_rate_worker']:.5f}/{hyperparams['epsilon_min_worker']:.3f}")
    print(
        f"INFO: Target/Sync Freq (Train Steps):{hyperparams['update_target_every_n_steps']}, Learn Freq (Global Env Steps):{hyperparams['learn_every_n_steps']}")
    print(
        f"INFO: ExpBatchWorkerSize: {hyperparams['experience_batch_send_size']}")

    session_start_time = time.time()
    # Para calcular SPS (Steps Per Second)
    # Usar atributos en la función para mantener estado entre llamadas implícitas por el loop
    if not hasattr(run_ai_training, 'last_print_time_attr'):
        run_ai_training.last_print_time_attr = session_start_time  # type: ignore
    if not hasattr(run_ai_training, 'last_global_step_print'):
        run_ai_training.last_global_step_print = current_global_step  # type: ignore

    training_manually_halted = False

    try:
        while current_global_episode <= num_total_episodes_target:
            try:
                # Obtener un batch de experiencias de un worker
                experience_batch_from_worker = experience_submission_q.get(
                    timeout=0.1)  # Timeout pequeño

                for exp_package_tuple in experience_batch_from_worker:
                    exp_state, exp_action, exp_reward, exp_next_state, exp_done, exp_score_if_done = exp_package_tuple
                    main_learner_agent.remember(
                        exp_state, exp_action, exp_reward, exp_next_state, exp_done)
                    current_global_step += 1

                    if exp_done:  # Un worker completó un episodio
                        # plot_scores_data_session es (ep_global, score)
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

                            print(f"EpG:{current_global_episode}/{num_total_episodes_target} StG:{current_global_step//1000}k ({sps_val:.0f}SPS) "
                                  f"LastS:{exp_score_if_done} EpsL:{main_learner_agent.epsilon:.3f} "
                                  f"AvgS(100):{avg_score_val:.2f} Mem:{len(main_learner_agent.memory)//1000}k TrainS:{num_training_steps_done//1000}k")
                            if show_plot_flag and live_plot_line_score is not None:
                                update_matplotlib_plot(
                                    plot_scores_data_session, live_plot_line_score, live_plot_line_avg_score)

                        current_global_episode += 1  # Incrementar contador de episodios globales

                        # Guardar checkpoint basado en episodios globales
                        if current_global_episode % SAVE_CHECKPOINT_EVERY_DEFAULT == 0 or \
                           current_global_episode > num_total_episodes_target:  # Guardar al final también
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

                # Lógica de entrenamiento (replay) del learner principal
                if len(main_learner_agent.memory) >= main_learner_agent.batch_size and \
                   current_global_step % hyperparams['learn_every_n_steps'] == 0:
                    main_learner_agent.replay()
                    num_training_steps_done += 1

                    # Decaimiento de epsilon del learner global
                    if main_learner_agent.epsilon > hyperparams['epsilon_min_global']:
                        main_learner_agent.epsilon *= hyperparams['epsilon_decay_rate_global']
                        main_learner_agent.epsilon = max(
                            hyperparams['epsilon_min_global'], main_learner_agent.epsilon)

                    # Actualizar target model y sincronizar pesos con workers
                    if num_training_steps_done > 0 and \
                       num_training_steps_done % hyperparams['update_target_every_n_steps'] == 0:
                        main_learner_agent.update_target_model()
                        # print(f"INFO: Target actualizado (TS:{num_training_steps_done}). Learner Eps:{main_learner_agent.epsilon:.3f}")

                        updated_weights_for_workers = main_learner_agent.model.get_weights()
                        for i in range(num_workers_to_use):
                            try:
                                # Limpiar cola antes de poner nuevos pesos
                                while not model_weights_broadcast_qs[i].empty():
                                    model_weights_broadcast_qs[i].get_nowait()
                                model_weights_broadcast_qs[i].put(
                                    updated_weights_for_workers, block=False)  # No bloquear
                            except queue.Full:
                                # print(f"WARN: Worker {i} weight queue full during sync. Skipping.")
                                pass  # Normal si el worker está ocupado
                            except Exception as e_q_put:
                                print(
                                    f"WARN: Error enviando pesos a worker {i}: {e_q_put}")

            except queue.Empty:  # Si no hay experiencias en la cola, esperar un poco
                # Pequeña pausa para no consumir CPU excesivamente
                time.sleep(0.001)
                pass
            except KeyboardInterrupt:
                print("\nINFO: Entrenamiento interrumpido manualmente.")
                training_manually_halted = True
                break
            except Exception as e_main_loop:
                print(
                    f"ERROR: Inesperado en el bucle principal: {e_main_loop}")
                import traceback
                traceback.print_exc()
                training_manually_halted = True  # Tratar como interrupción para guardar
                break

        if current_global_episode > num_total_episodes_target and not training_manually_halted:
            print("\nINFO: Número de episodios objetivo alcanzado.")

    finally:  # Limpieza y guardado final
        print("\nINFO: Finalizando sesión de entrenamiento...")
        stop_workers_event.set()  # Señal para que los workers terminen

        for i, p_worker_proc in enumerate(active_workers_list):
            # Dar tiempo para que terminen limpiamente
            p_worker_proc.join(timeout=10)
            if p_worker_proc.is_alive():
                print(
                    f"WARN: Worker {i} no terminó a tiempo, forzando terminación...")
                p_worker_proc.terminate()  # Forzar si no responde
                p_worker_proc.join()      # Esperar a que termine la terminación forzada
        print(
            f"INFO: Todos los {len(active_workers_list)} workers han sido señalados para terminar y se ha esperado su unión.")

        # Cerrar colas de forma segura
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
            f"INFO: Sesión de entrenamiento {'detenida manualmente' if training_manually_halted else 'finalizada'}.")

        # El último episodio global que se completó (o estaba en curso si se interrumpió)
        final_completed_episode_num = current_global_episode - 1
        print(
            f"INFO: Último episodio global registrado/completado: {final_completed_episode_num}")
        print(
            f"INFO: Duración total de ESTA SESIÓN: {total_duration_minutes:.2f} minutos.")

        steps_this_session = current_global_step - initial_global_step_for_session
        train_steps_this_session = num_training_steps_done - initial_train_steps_for_session
        print(
            f"INFO: Pasos globales de entorno procesados en esta sesión: {steps_this_session}")
        print(
            f"INFO: Pasos de entrenamiento (replay) realizados en esta sesión: {train_steps_this_session}")

        if hasattr(main_learner_agent, 'model') and main_learner_agent.model and final_completed_episode_num >= 0:
            print(
                f"INFO: Guardando estado final en episodio global {final_completed_episode_num}...")
            main_learner_agent.save_keras_model()
            final_training_state_to_save = {
                'last_completed_episode_global': final_completed_episode_num,
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
                print(
                    f"ERROR: Al guardar el estado final del entrenamiento: {e_final_save}")

        if show_plot_flag and fig_plot is not None:  # type: ignore
            plt.ioff()  # type: ignore
            try:
                if plt.fignum_exists(fig_plot.number):  # type: ignore
                    plot_save_filename = f"dqn_training_plot_{model_base_filename}_{AGENT_VERSION_TRAINER}_ep{final_completed_episode_num}.png"
                    plot_save_path = os.path.join(DATA_DIR, plot_save_filename)
                    fig_plot.savefig(plot_save_path)  # type: ignore
                    print(
                        f"INFO: Gráfica de entrenamiento guardada como {plot_save_path}")
                    plt.close(fig_plot)  # type: ignore
            except Exception as e_plot_save:
                print(
                    f"WARN: Error guardando/cerrando gráfica de entrenamiento: {e_plot_save}")

        print("INFO: Proceso principal de entrenamiento finalizado.")


def play_mode_with_ui(model_keras_path_full, num_episodes_to_play, play_speed_delay):
    """Ejecuta el agente en modo demostración con UI de Arcade (snake_ui_v11)."""
    try:
        import arcade  # type: ignore
        from snake_ui_v11 import SnakeGameUI  # Usar la UI de v11 como en trainer_v6.py
    except ImportError as e_arcade:
        print(
            f"ERROR: Al importar Arcade o SnakeGameUI (v11): {e_arcade}. Saliendo del modo play.")
        return

    print(
        f"\n--- Viendo jugar al agente DQN v{AGENT_VERSION_TRAINER} ({model_keras_path_full}) con UI ---")
    if not os.path.exists(model_keras_path_full):
        print(
            f"ERROR: No se encontró el modelo Keras en '{model_keras_path_full}'. Saliendo.")
        return

    # Usar la lógica de juego v12 para la demostración
    game_logic_instance = SnakeLogic(
        SCREEN_WIDTH_LOGIC, SCREEN_HEIGHT_LOGIC, STEP_SIZE_LOGIC)

    # Agente para jugar (epsilon muy bajo, state_size=12)
    # El agente se instancia desde agent_v6_parallel
    agent_for_play = DQNAgent(
        state_size=STATE_SIZE, action_size=ACTION_SIZE,  # STATE_SIZE es 12
        epsilon=0.001, epsilon_min=0.001,  # Epsilon muy bajo para explotación
        model_filepath=model_keras_path_full,
        learning_rate=0.00001  # LR no es crítico para jugar
    )
    if not agent_for_play.model:  # Verificar si el modelo se cargó
        print(
            f"ERROR: No se pudo cargar el modelo del agente desde {model_keras_path_full}. Saliendo.")
        return

    ui_instance_obj = None
    try:
        print(f"INFO: Iniciando UI (SnakeGameUI de snake_ui_v11) para demostración.")
        ui_instance_obj = SnakeGameUI(
            # Pasar la instancia de SnakeLogic v12
            snake_logic_instance=game_logic_instance,
            agent_instance=agent_for_play,
            num_demo_episodes=num_episodes_to_play,
            play_speed=play_speed_delay,
            ai_controlled_demo=True
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
                arcade.exit()  # Intenta cerrar Arcade limpiamente
            except Exception:
                pass  # Ignorar errores si ya está cerrado


if __name__ == "__main__":
    # Configuración del método de inicio para multiprocessing (importante para algunos OS)
    try:
        current_mp_context = mp.get_context()
        # No establecido
        if current_mp_context.get_start_method(allow_none=True) is None:
            # 'spawn' es más seguro y compatible
            mp.set_start_method('spawn', force=False)
            print("INFO: Método de inicio de multiprocessing establecido en 'spawn'.")
        elif current_mp_context.get_start_method() != 'spawn':
            try:  # Intentar forzar si es diferente y no es None
                mp.set_start_method('spawn', force=True)
                print("INFO: Método de inicio de multiprocessing forzado a 'spawn'.")
            except RuntimeError as e_mp_force_spawn:
                print(
                    f"WARN: No se pudo forzar mp a 'spawn': {e_mp_force_spawn}. Usando: {current_mp_context.get_start_method()}")
        # else: print(f"INFO: Método de inicio de multiprocessing ya es '{current_mp_context.get_start_method()}'.")
    except Exception as e_set_mp_method:
        print(
            f"WARN: Excepción estableciendo método de inicio de multiprocessing: {e_set_mp_method}")

    mp.freeze_support()  # Necesario para ejecutables Windows

    setup_tensorflow_main_process()
    setup_matplotlib_main_process()

    parser = argparse.ArgumentParser(
        description=f"Entrenar o ver IA Snake (v{AGENT_VERSION_TRAINER} - DQN Seq + MP, StateSize={STATE_SIZE})")
    parser.add_argument("--play", action="store_true",
                        help="Ver jugar al agente.")
    parser.add_argument("--reset", action="store_true",
                        help="Reiniciar entrenamiento eliminando modelo y checkpoint.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT,
                        help=f"Episodios globales objetivo (def: {NUM_EPISODES_DEFAULT}).")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS_DEFAULT,
                        help=f"Número de procesos worker (def: {NUM_WORKERS_DEFAULT}).")
    parser.add_argument("--plot", action="store_true",
                        help="Mostrar gráfica Matplotlib.")

    # Hiperparámetros del Learner y Worker
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_DQN_DEFAULT,
                        help="Tasa de aprendizaje (def: %(default)s).")
    parser.add_argument("--gamma", type=float, default=DISCOUNT_FACTOR_DQN_DEFAULT,
                        help="Factor de descuento (def: %(default)s).")

    parser.add_argument("--epsilon_start", type=float, default=EPSILON_START_DQN_DEFAULT,
                        help="Epsilon inicial (global y worker) (def: %(default)s).")
    parser.add_argument("--epsilon_decay_global", type=float, default=EPSILON_DECAY_RATE_GLOBAL_DEFAULT,
                        help="Decaimiento epsilon learner (def: %(default)s).")
    parser.add_argument("--epsilon_min_global", type=float, default=EPSILON_MIN_GLOBAL_DEFAULT,
                        help="Mínimo epsilon learner (def: %(default)s).")

    parser.add_argument("--epsilon_decay_worker", type=float, default=EPSILON_DECAY_RATE_WORKER_DEFAULT,
                        help="Decaimiento epsilon worker (def: %(default)s).")
    parser.add_argument("--epsilon_min_worker", type=float, default=EPSILON_MIN_WORKER_DEFAULT,
                        help="Mínimo epsilon worker (def: %(default)s).")

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DQN_DEFAULT,
                        help="Tamaño de batch para replay (def: %(default)s).")
    parser.add_argument("--replay_memory", type=int, default=REPLAY_MEMORY_SIZE_DQN_DEFAULT,
                        help="Capacidad memoria replay (def: %(default)s).")
    parser.add_argument("--agent_epochs", type=int, default=AGENT_EPOCHS_PER_REPLAY_DEFAULT,
                        help="Epochs por replay() (def: %(default)s).")

    parser.add_argument("--update_target_train_steps", type=int, default=UPDATE_TARGET_AND_SYNC_WORKERS_EVERY_N_TRAIN_STEPS,
                        help="Pasos de entrenamiento para actualizar target y sync workers (def: %(default)s).")
    parser.add_argument("--learn_global_steps", type=int, default=LEARN_EVERY_N_GLOBAL_STEPS,
                        help="Pasos de entorno globales para un replay() (def: %(default)s).")
    parser.add_argument("--exp_batch_send_size", type=int, default=EXPERIENCE_BATCH_FROM_WORKER_SIZE,
                        help="Tamaño batch experiencias de worker a learner (def: %(default)s).")

    parser.add_argument("--model_file_base", type=str, default=MODEL_FILENAME_BASE_DEFAULT,
                        help="Nombre base para archivos (def: %(default)s).")

    # Argumentos para el modo Play
    parser.add_argument("--play-episodes", type=int, default=5,
                        help="Episodios para ver en modo play (def: 5).")
    parser.add_argument("--play-speed", type=float, default=0.05,
                        help="Velocidad UI en modo play (segundos/paso) (def: 0.05).")

    args = parser.parse_args()

    if not args.play and args.num_workers <= 0:
        print(
            f"WARN: num_workers ({args.num_workers}) inválido para entrenamiento. Usando 1 worker.")
        args.num_workers = 1
    elif args.num_workers > mp.cpu_count() and not args.play:  # mp.cpu_count() * 2 puede ser demasiado
        print(
            f"WARN: num_workers ({args.num_workers}) podría ser alto para {mp.cpu_count()} CPUs. Considere reducirlo.")

    current_hyperparams_dict = {
        'lr': args.lr, 'gamma': args.gamma,
        'initial_epsilon_global': args.epsilon_start,  # Mismo inicio para learner
        'epsilon_decay_rate_global': args.epsilon_decay_global,
        'epsilon_min_global': args.epsilon_min_global,

        'initial_epsilon_worker': args.epsilon_start,  # Mismo inicio para worker
        'epsilon_decay_rate_worker': args.epsilon_decay_worker,
        'epsilon_min_worker': args.epsilon_min_worker,

        'batch_size': args.batch_size,
        'replay_memory': args.replay_memory,
        'agent_epochs': args.agent_epochs,

        'update_target_every_n_steps': args.update_target_train_steps,
        'learn_every_n_steps': args.learn_global_steps,
        'experience_batch_send_size': args.exp_batch_send_size,

        'model_file_base_name': args.model_file_base
    }

    if args.play:
        model_to_play_path = os.path.join(
            DATA_DIR, f"{args.model_file_base}_{AGENT_VERSION_TRAINER}.keras")
        play_mode_with_ui(
            model_keras_path_full=model_to_play_path,
            num_episodes_to_play=args.play_episodes,
            play_speed_delay=args.play_speed
        )
    else:
        # Inicializar atributos de función para estadísticas de SPS si no existen
        if not hasattr(run_ai_training, 'last_print_time_attr'):
            run_ai_training.last_print_time_attr = time.time()  # type: ignore
        if not hasattr(run_ai_training, 'last_global_step_print'):
            run_ai_training.last_global_step_print = 0  # type: ignore

        run_ai_training(
            num_total_episodes_target=args.episodes, reset_model_flag=args.reset,
            hyperparams=current_hyperparams_dict, show_plot_flag=args.plot,
            num_workers_to_use=args.num_workers
        )
