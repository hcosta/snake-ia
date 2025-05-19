# How to

La diferencia entre la versión 1 y 2 es que en la segunda se ha separado la muerte por colisión, ya no es una global sino que se diferencia entre chocar contra paredes o la propia serpiente, añadiendo unas recompensas/castigos especificos de cada caso:

* Ejecutar una prueba con los datos entrenados:
  
```bash
pipenv run python trainer_v1.py --play
pipenv run python trainer_v2.py --play
```

* Seguir entrenando los datos:
  
```bash
pipenv run python trainer_v1.py --no-visualize --episodes=10000
pipenv run python trainer_v2.py ---no-visualize --episodes=10000
```

* Reiniciar el entrenamiento:
  
```bash
pipenv run python trainer_v1.py --reset
pipenv run python trainer_v2.py --reset
```

## Parámetros disponibles

Estos son los argumentos que puedes usar al ejecutar el script para controlar su comportamiento:

- `--play`  
  Ver jugar al agente utilizando la Q-Table previamente entrenada.

- `--episodes <int>`  
  Número de episodios para entrenar.  
  **Ejemplo:** `--episodes 1000` (por defecto: `NUM_EPISODES_DEFAULT`).

- `--no-visualize`  
  Desactiva la visualización gráfica del juego durante el entrenamiento.  
  **Nota:** este parámetro desactiva el valor por defecto `--visualize`.

- `--play-episodes <int>`  
  Número de partidas que se reproducirán al ejecutar `--play`.  
  **Ejemplo:** `--play-episodes 10` (por defecto: 5).

- `--play-speed <float>`  
  Tiempo de espera entre frames (en segundos) al ver jugar al agente.  
  **Ejemplo:** `--play-speed 0.08`.

- `--reset`  
  Elimina la Q-Table guardada, reiniciando el entrenamiento desde cero.

- `--visualization-speed <float>`  
  Tiempo de pausa entre frames durante la visualización del entrenamiento.  
  **Ejemplo:** `--visualization-speed 0.1` (por defecto: `VISUALIZATION_UPDATE_RATE`).  
  Usa `0` para la máxima velocidad.