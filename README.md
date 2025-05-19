# Snake IA (Q-Learning vs Deep Q-Learning)

> Un día me aburría y entrené una IA (para jugar Snake). ¡Y la cosa se puso interesante!

<table>
  <tr>
    <td align="center">Q-Learning en Acción</td>
    <td align="center">Deep Q-Learning en Acción</td>
  </tr>
  <tr>
    <td><img src="./docs/demo-QL.gif" alt="Demostración del Agente Q-Learning"/></td>
    <td><img src="./docs/demo-DQN.gif" alt="Demostración del Agente Deep Q-Learning"/></td>
  </tr>
</table>

## 🎯 El Proyecto

Este proyecto nació de la curiosidad y el desafío personal de entrenar una Inteligencia Artificial para dominar el clásico juego de Snake. Para ello, he explorado y comparado dos enfoques principales del aprendizaje por refuerzo:

1.  **Primer Intento: Aprendizaje con una "Chuleta" (Q-Learning Clásico)**
2.  **Segundo Intento: Un "Cerebro" para la Serpiente (Deep Q-Learning con Redes Neuronales)**

El objetivo final es observar la evolución del aprendizaje, las limitaciones de cada método y, por supuesto, ¡ver hasta dónde puede llegar cada IA!

## 🤔 ¿En qué consiste esta práctica? (La explicación para Dummies)

Imagina que queremos enseñarle a una serpiente digital a jugar al Snake.

1.  **El Método de la "Chuleta" (Q-Learning Clásico)**:
    * Al principio, es como si le diéramos a la serpiente una enorme hoja de trucos. Por cada situación posible en el juego (dónde está la comida, dónde están los obstáculos, etc.), la IA anota qué tan buena o mala es cada posible acción (moverse arriba, abajo, izquierda o derecha).
    * Fui ajustando cómo la IA "veía" el tablero y qué consideraba un "premio" (comer la fruta) o un "castigo" (chocar). También le enseñé a reconocer peligros básicos, como meterse en un callejón sin salida.
    * Aunque la serpiente aprendió bastante, llegó un punto en que no mejoraba más. Se quedaba atascada en sus propios enredos o no sabía cómo salir de situaciones complicadas. La "chuleta" se volvía demasiado grande para todas las posibilidades del juego.

2.  **El Método del "Cerebro Artificial" (Deep Q-Learning con Redes Neuronales)**:
    * Como la "chuleta" no fue suficiente, decidí darle a la serpiente un pequeño "cerebro" artificial. Este cerebro está hecho con redes neuronales (usando una herramienta popular llamada TensorFlow) y es mucho más listo. Puede aprender patrones complejos y tomar decisiones inteligentes, incluso en situaciones que no ha visto antes de forma idéntica.
    * Este "cerebro" aprende jugando muchísimas partidas. Para que aprenda bien, usé algunas técnicas especiales:
        * Tiene una **memoria** donde guarda lo que hizo y qué pasó después, para poder repasar y aprender de sus aciertos y errores.
        * Tiene una especie de **"copia de seguridad" de sí mismo** que se actualiza más despacio, ayudando a que el aprendizaje sea más estable.
    * También reorganicé el código del juego para poder entrenar a la IA sin necesidad de verla jugar todo el tiempo (lo que se llama entrenamiento "headless", que es mucho más rápido) y para poder tener diferentes formas de verla jugar (una con gráficos y otra en modo texto).
    * ¡Y las sensaciones son muy buenas! Este "cerebro" parece estar aprendiendo a no quedarse atrapado tan fácilmente y a desarrollar estrategias más astutas.

## 📊 Experiencia y Observaciones

### Q-Learning Clásico ("La Chuleta")

* **Entrenamiento y Rendimiento**: Después de un entrenamiento exhaustivo de aproximadamente 200,000 episodios (unas 4 horas de procesamiento), el agente de Q-Learning logró estabilizarse con una **puntuación media de alrededor de 25-27 puntos**. En ocasiones puntuales, podía alcanzar puntuaciones significativamente más altas (picos cercanos a 40).
* **Desafíos Iniciales**: Un primer obstáculo fue ajustar la exploración. Si el agente exploraba demasiado durante mucho tiempo, sus movimientos aleatorios le impedían superar los 15 puntos. Reducir la tasa de exploración a largo plazo fue clave para mejorar.
* **El Gran Límite**: El principal problema insuperable para este modelo fue su incapacidad para resolver eficazmente el problema de **quedar encerrado en sí mismo**. La serpiente aprendía a evitar las paredes y a buscar la comida, pero la planificación a largo plazo para no crear bucles fatales resultaba demasiado compleja para la tabla Q.
* **Conclusión sobre Q-Learning**: Aunque el juego *Snake* parece simple inicialmente (ir hacia la comida), requiere una estrategia considerable a largo plazo para la supervivencia. Los sistemas Q-Learning clásicos, al depender de memorizar cada estado, no parecen ser la herramienta más adecuada para dominar esta complejidad inherente, lo que me motivó a explorar las redes neuronales.

<p align="center">
  <img src="./docs/demo-QL-Log.gif" alt="isualización del agente QL en modo juego (sin entrenamiento)"/>
  <em>Visualización del agente QL en modo juego (sin entrenamiento).</em>
</p>

### Deep Q-Learning ("El Cerebro Artificial")

* **Retos Técnicos Iniciales**:
    * **Configuración de TensorFlow y GPU**: El primer escollo fue el rendimiento. Entrenar la red neuronal usando solo la CPU era extremadamente lento. Intenté configurar TensorFlow para usar la GPU (con CUDA) de forma nativa en Windows, pero no fue sencillo. La solución vino al utilizar contenedores Docker con entornos de TensorFlow preconfigurados por Nvidia, lo que permitió acelerar significativamente el proceso.
    * **Entrenamiento Headless**: Los contenedores Docker, por defecto, carecen de entorno gráfico, lo que impedía ejecutar la visualización del juego. Aunque inicialmente usé el comando `xvfb-run` como una solución temporal para simular un entorno gráfico, finalmente opté por una solución más robusta: refactorizar el código para separar completamente la lógica del juego de la interfaz gráfica. Esto permitió un entrenamiento "headless" eficiente y la creación de múltiples interfaces (una con Arcade y otra en modo texto para la terminal, ¡gracias `Gemini` por la ayuda con la versión `curses`!).
* **Evolución del Aprendizaje**:
    * Tras aproximadamente 8,500 episodios de entrenamiento (unas 10-12 horas de procesamiento con GPU, aunque con cierto cuello de botella debido a la naturaleza de Python y la comunicación con la GPU), el agente DQN está mostrando un rendimiento muy prometedor. En las últimas fases del entrenamiento, está promediando de forma estable alrededor de **15-17 puntos** (media de los últimos 100 episodios), con picos individuales que superan los 30 puntos (¡e incluso llegando a 41 en pruebas!).
    * Es importante destacar que, aunque la media de entrenamiento actual del DQN es inferior a la media final del Q-Learning, el DQN lo ha logrado con **muchísimos menos episodios de entrenamiento** (8.5k vs 200k) y sigue mostrando una clara tendencia ascendente.
    * El aprendizaje sigue un patrón de "dientes de sierra": hay ciclos de mejora, seguidos de pequeñas bajadas o mesetas donde parece consolidar lo aprendido, para luego volver a escalar y superar el rendimiento anterior. Por ejemplo, tardó unos 3,500 episodios en promediar consistentemente 5 puntos, pero la progresión se ha acelerado notablemente después.
* **Comparativa Actual y Próximos Pasos**:
    * Aunque el Q-Learning, con su vasto entrenamiento, todavía puede mostrar una media ligeramente superior en tandas de prueba cortas, el **potencial y la eficiencia de aprendizaje del DQN son claramente superiores**. Ya iguala e incluso supera los picos de rendimiento del Q-Learning con una fracción del entrenamiento.
    * El principal desafío para el DQN sigue siendo perfeccionar las estrategias para evitar auto-colisiones a largo plazo. Continuaré el entrenamiento hasta los 100,000 episodios (o más si sigue mejorando) para observar si puede superar consistentemente al Q-Learning y dominar este aspecto del juego. No descarto reiniciar el entrenamiento con hiperparámetros ajustados si se observa un estancamiento persistente más adelante, pero por ahora, ¡la progresión es alentadora!

<p align="center">
  <img src="./docs/shell-DQN.png" alt="Logs del entrenamiento inicial"/>
  <em>Logs del entrenamiento tras 8.500 episodios.</em>
</p>

<p align="center">
  <img src="./docs/plot-DQN.png" alt="Gráfico de recompensas del entrenamiento DQN" width="70%"/>
  <em>Curva de aprendizaje del agente DQN.</em>
</p>

<p align="center">
  <img src="./docs/demo-DQN-Log.gif" alt="Visualización del agente DQN en modo juego (sin entrenamiento)"/>
  <em>Visualización del agente DQN en modo juego (sin entrenamiento).</em>
</p>

## 🚀 Cómo Probar los Agentes

* **Instalar las dependencias en un entorno virtual, se requiere python 3.11 o editar `python_version` del `Pipfile` manualmente con la versión del sistema**:
    ```bash
    pip install pipenv  # si no lo tienes
    pipenv install
    ```
* **Ejecutar el Agente Q-Learning**:
    ```bash
    cd snake-QL
    pipenv run python trainer_v2.py --play --play-speed=0.02
    ```
* **Ejecutar el Agente DQN**:
    ```bash
    cd snake-DQN
    pipenv run python trainer_v6_parallel.py --play --play-speed=0.02
    ```
* **Jugar al Snake sin IA en ventana (con Python Arcade)**:
    ```bash
    cd snake-DQN
    pipenv run python snake_ui_v10.py  
    ```
* **Jugar al Snake sin IA en la shell (con módulo Curses)**:
    ```bash
    cd snake-DQN
    pipenv run python snake_shell_v10.py  
    ```