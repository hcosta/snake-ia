# trainer_v3.py (Entrenador para DQN)
import matplotlib.pyplot as plt
from agent_v3 import DQNAgent  # usamos el agente optimizad para la GPU
from snake_v9 import SnakeGame, SCREEN_WIDTH, SCREEN_HEIGHT, STEP_SIZE, POINTS_PER_FOOD, REWARD_STUCK_LOOP
import arcade
import numpy as np
import time
import os
import argparse
import tensorflow as tf

# HABILITAR CRECIMIENTO DE MEMORIA PARA LA GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        # El crecimiento de memoria debe establecerse antes de que las GPUs se hayan inicializado
        print(f"Error al habilitar crecimiento de memoria: {e}")
else:
    print("No se detectaron GPUs por TensorFlow.")

# Asumiendo que snake_v10.py es igual a snake_v9.py o usas snake_v9.py directamente
# Importar el nuevo agente DQN

# --- Constantes de Entrenamiento ---
# El estado de snake_v9.py tiene 9 características
STATE_SIZE = 9
# 4 acciones: Arriba, Abajo, Izquierda, Derecha (índices 0, 1, 2, 3)
ACTION_SIZE = 4
POSSIBLE_ACTIONS_INDICES = list(range(ACTION_SIZE))  # [0, 1, 2, 3]

# --- Hiperparámetros del Agente DQN (Ejemplos, necesitan ajuste) ---
LEARNING_RATE_DQN = 0.001        # Tasa de aprendizaje para Adam optimizer
DISCOUNT_FACTOR_DQN = 0.975       # Factor de descuento para recompensas futuras
EPSILON_START_DQN = 1.0          # Epsilon inicial para exploración
EPSILON_DECAY_DQN = 0.9995        # Tasa de decaimiento de epsilon por episodio
# Un decaimiento más lento (ej. 0.9995 o 0.9999) puede ser mejor
EPSILON_MIN_DQN = 0.01           # Epsilon mínimo

REPLAY_MEMORY_SIZE_DQN = 20000   # Tamaño máximo de la memoria de repetición
# Aumentar puede ayudar pero consume más RAM
BATCH_SIZE_DQN = 64  # PARA LA GPU o 64 para CPU
# Comunes: 32, 64, 128
# Con qué frecuencia (en episodios) actualizar el target_model
UPDATE_TARGET_EVERY = 5
# También se puede hacer por número de pasos

MODEL_FILENAME = "dqn_snake_model_v1.weights.h5"

# --- Parámetros de Entrenamiento del Trainer ---
# DQN necesita muchos episodios, empezar con menos para probar
NUM_EPISODES_DEFAULT = 1000
# Luego aumentar a 5000, 10000, o más.
VISUALIZE_TRAINING_DEFAULT = True
# Más rápido para DQN si no se visualiza cada paso
VISUALIZATION_UPDATE_RATE = 0.01
SAVE_MODEL_EVERY = 100
PRINT_STATS_EVERY = 10

# --- Matplotlib Setup (igual que antes) ---
plt.ion()
fig_plot, ax_plot = None, None
plot_scores = []
plot_avg_scores = []


def setup_matplotlib_plot():
    global fig_plot, ax_plot
    fig_plot, ax_plot = plt.subplots(figsize=(10, 5))
    ax_plot.set_title("Resultados del Entrenamiento de Snake IA (DQN)")
    ax_plot.set_xlabel("Episodio")
    ax_plot.set_ylabel("Puntuación")
    line_score, = ax_plot.plot([], [], 'b-', label='Puntuación Episodio')
    line_avg_score, = ax_plot.plot(
        [], [], 'r-', label='Media Puntuación')
    ax_plot.legend()
    return line_score, line_avg_score


def update_matplotlib_plot(scores_history_for_plot, line_score, line_avg_score):
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


def run_ai_training(num_episodes, visualize_training, visualization_update_rate, reset_model_flag):
    global plot_scores
    plot_scores = []  # Reiniciar para la gráfica

    env = None
    if reset_model_flag and os.path.exists(MODEL_FILENAME):
        try:
            os.remove(MODEL_FILENAME)
            print(
                f"Se ha eliminado '{MODEL_FILENAME}' para reiniciar el entrenamiento.")
        except OSError as e:
            print(f"Error al eliminar '{MODEL_FILENAME}': {e}")

    env_title = "Snake IA (DQN) - Entrenamiento"
    env = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT, env_title, ai_controlled=True)

    agent = DQNAgent(state_size=STATE_SIZE,
                     action_size=ACTION_SIZE,
                     learning_rate=LEARNING_RATE_DQN,
                     discount_factor=DISCOUNT_FACTOR_DQN,
                     epsilon=EPSILON_START_DQN,
                     epsilon_decay=EPSILON_DECAY_DQN,  # Decay se maneja en replay()
                     epsilon_min=EPSILON_MIN_DQN,
                     replay_memory_size=REPLAY_MEMORY_SIZE_DQN,
                     batch_size=BATCH_SIZE_DQN,
                     model_filepath=MODEL_FILENAME)

    if reset_model_flag:  # Si se resetea, epsilon también debería empezar desde el inicio
        agent.reset_epsilon(EPSILON_START_DQN)
        print(f"Epsilon reiniciado a: {agent.epsilon:.4f} debido a --reset.")

    scores_history_session = []
    total_steps_session = 0
    training_manually_stopped = False

    line_score, line_avg_score = None, None
    if visualize_training:  # Solo configurar matplotlib si se va a visualizar
        line_score, line_avg_score = setup_matplotlib_plot()

    print(f"Iniciando entrenamiento DQN para {num_episodes} episodios...")
    print(
        f"Hiperparámetros DQN: LR={agent.learning_rate}, Gamma={agent.gamma}, EpsilonDecay={agent.epsilon_decay}, Batch={agent.batch_size}")
    start_time_total = time.time()

    try:
        for episode_idx in range(1, num_episodes + 1):
            current_state = env.reset()  # Obtiene el estado inicial (tupla de 9 números)
            # Reformatear para la red
            current_state = np.reshape(current_state, [1, STATE_SIZE])

            episode_score = 0
            episode_reward_sum = 0
            done = False
            steps_in_episode = 0
            # La detección de "stuck loop" de snake_v9.py sigue siendo útil
            steps_since_last_food = 0
            last_score_for_food_check = 0

            while not done:
                if visualize_training:
                    env.dispatch_events()
                    if env.training_should_stop:
                        training_manually_stopped = True
                        break

                # Devuelve el índice de la acción (0,1,2,3)
                action_idx = agent.get_action(current_state)

                # El entorno step() espera la acción directa (0,1,2,3)
                next_state_tuple, reward, done, info = env.step(action_idx)
                next_state = np.reshape(next_state_tuple, [1, STATE_SIZE])

                # Modificar la recompensa si se queda atascado (lógica de snake_v9)
                # Esta lógica podría estar dentro de env.step o aquí
                if env.score > last_score_for_food_check:
                    steps_since_last_food = 0
                    last_score_for_food_check = env.score
                else:
                    steps_since_last_food += 1

                max_steps_no_food_limit = 60  # O la que uses en snake_v9
                if not info.get('ate_food', False) and steps_since_last_food > max_steps_no_food_limit:
                    # Usar REWARD_STUCK_LOOP de snake_v9 si está definido o definirlo aquí
                    reward += REWARD_STUCK_LOOP  # Acceder a la constante del entorno
                    done = True  # Forzar fin del episodio
                    env.game_over = True  # Para que el entorno sepa que terminó
                    if 'collision_type' not in info or info['collision_type'] is None:
                        info['collision_type'] = 'stuck'

                episode_reward_sum += reward
                episode_score = env.score

                # Guardar en memoria de repetición
                # current_state ya está reformateado. next_state también.
                # action_idx es el índice de la acción.
                agent.remember(
                    current_state[0], action_idx, reward, next_state[0], done)

                current_state = next_state
                steps_in_episode += 1
                total_steps_session += 1

                # Entrenar el agente (replay)
                # Se puede hacer cada N pasos en lugar de cada paso si es muy lento
                if len(agent.memory) > agent.batch_size:  # Solo si hay suficientes muestras
                    agent.replay()

                if visualize_training:
                    env.clear()
                    env.on_draw()  # Dibuja el estado actual del juego
                    arcade.finish_render()
                    if done and episode_idx % PRINT_STATS_EVERY != 0:  # Pausa breve al final si no se va a imprimir
                        time.sleep(0.2)
                    else:
                        if visualization_update_rate > 0:
                            time.sleep(visualization_update_rate)
                if done:
                    break
            # --- Fin del bucle de pasos (while not done) ---

            if training_manually_stopped:
                break

            scores_history_session.append(episode_score)
            plot_scores.append(episode_score)  # Para la gráfica global

            # Actualizar el modelo objetivo (target network) cada N episodios
            if episode_idx % UPDATE_TARGET_EVERY == 0:
                agent.update_target_model()
                # print(f"Episodio {episode_idx}: Modelo objetivo actualizado.")

            if episode_idx % PRINT_STATS_EVERY == 0 or episode_idx == num_episodes:
                avg_score_last_100 = np.mean(
                    scores_history_session[-100:]) if scores_history_session else 0
                print(f"Ep: {episode_idx}/{num_episodes} | Steps: {steps_in_episode} | Score: {episode_score} | "
                      f"Total Reward: {episode_reward_sum:.1f} | Epsilon: {agent.epsilon:.4f} | "
                      f"Avg Score (last 100): {avg_score_last_100:.2f} | Memory: {len(agent.memory)}")
                if visualize_training and line_score:
                    update_matplotlib_plot(
                        plot_scores, line_score, line_avg_score)

            if episode_idx % SAVE_MODEL_EVERY == 0 or episode_idx == num_episodes or training_manually_stopped:
                agent.save_model(MODEL_FILENAME)

            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay  # O usa EPSILON_DECAY_DQN directamente

        # --- Fin del bucle de episodios ---

    except KeyboardInterrupt:  # Permitir interrupción manual con Ctrl+C
        print("\nEntrenamiento interrumpido por el usuario (KeyboardInterrupt).")
        training_manually_stopped = True
    except Exception as e:
        print(f"Ocurrió un error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time_total = time.time()
        completed_episodes_count = episode_idx - \
            1 if training_manually_stopped and episode_idx > 0 else episode_idx

        print(
            f"\nEntrenamiento {'detenido' if training_manually_stopped else 'finalizado'}.")
        print(
            f"Se completaron {completed_episodes_count} episodios en esta sesión.")
        print(
            f"Duración total de la sesión: {((end_time_total - start_time_total)/60):.2f} minutos.")
        print(f"Total de pasos ejecutados en la sesión: {total_steps_session}")

        agent.save_model(MODEL_FILENAME)  # Guardar el modelo final

        if env:
            env.close()
        if fig_plot:
            plt.ioff()
            plt.savefig("dqn_training_plot.png")  # Guardar la gráfica
            print("Gráfica de entrenamiento guardada como dqn_training_plot.png")
            plt.close(fig_plot)


def play_with_trained_agent(model_path=MODEL_FILENAME, num_episodes=5, visualization_speed=0.08):
    print(f"\n--- Viendo jugar al agente DQN entrenado ({model_path}) ---")
    if not os.path.exists(model_path):
        print(
            f"Error: No se encontró el archivo del modelo en '{model_path}'.")
        print("Por favor, entrena al agente primero o verifica la ruta.")
        return

    env_play = SnakeGame(SCREEN_WIDTH, SCREEN_HEIGHT,
                         "Snake IA (DQN) - Demostración", ai_controlled=True)
    # Para jugar, epsilon debe ser muy bajo (casi sin exploración)
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
                     epsilon=0.00, epsilon_min=0.00, model_filepath=model_path)

    # Doble chequeo por si el constructor no pudo cargar
    if not os.path.exists(model_path):
        return

    for episode in range(1, num_episodes + 1):
        current_state_tuple = env_play.reset()
        current_state = np.reshape(current_state_tuple, [1, STATE_SIZE])
        env_play.training_should_stop = False  # Resetear bandera de parada
        done = False
        episode_score = 0
        print(f"\nIniciando demostración DQN - Episodio {episode}")

        while not done:
            env_play.dispatch_events()
            if env_play.training_should_stop:
                print("Saliendo de la demostración por petición del usuario.")
                if env_play:
                    env_play.close()
                    return

            action_idx = agent.get_action(current_state)  # Usará epsilon bajo
            next_state_tuple, reward, done, info = env_play.step(action_idx)
            next_state = np.reshape(next_state_tuple, [1, STATE_SIZE])

            current_state = next_state
            episode_score = env_play.score

            env_play.clear()
            env_play.on_draw()
            arcade.finish_render()
            time.sleep(visualization_speed)

            if done:
                print(
                    f"Episodio {episode} finalizado. Puntuación: {episode_score}. Colisión: {info.get('collision_type', 'N/A')}")
                time.sleep(1)  # Pausa para ver el game over
                break
    if env_play:
        env_play.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenar o ver jugar a la IA de Snake con DQN.")
    parser.add_argument("--play", action="store_true",
                        help="Ver jugar al agente con el modelo guardado.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_DEFAULT,
                        help=f"Número de episodios para entrenar (default: {NUM_EPISODES_DEFAULT}).")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                        help="Desactivar visualización Arcade durante entrenamiento.")
    parser.add_argument("--play-episodes", type=int, default=5,
                        help="Número de episodios para ver jugar al agente (default: 5).")
    parser.add_argument("--play-speed", type=float, default=0.05,
                        help="Segundos entre frames al ver jugar (ej: 0.05).")
    parser.add_argument("--reset", action="store_true",
                        help="Reiniciar entrenamiento eliminando el modelo guardado.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE_DQN,
                        help="Tasa de aprendizaje para DQN.")
    # Añadir más argumentos para otros hiperparámetros si se desea

    parser.set_defaults(visualize=VISUALIZE_TRAINING_DEFAULT)
    args = parser.parse_args()

    # Actualizar hiperparámetros globales si se pasan por CLI
    LEARNING_RATE_DQN = args.lr
    # ... (hacer lo mismo para otros hiperparámetros si se añaden al parser)

    if args.play:
        play_with_trained_agent(model_path=MODEL_FILENAME,
                                num_episodes=args.play_episodes,
                                visualization_speed=args.play_speed)
    else:
        run_ai_training(num_episodes=args.episodes,
                        visualize_training=args.visualize,
                        # Podrías hacerlo configurable también
                        visualization_update_rate=VISUALIZATION_UPDATE_RATE,
                        reset_model_flag=args.reset)
