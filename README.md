# Actividad Parcial 2 · Sección 5 — Búsqueda con heurísticas (A* vs Dijkstra)

Fundamentos de Inteligencia Artificial · Universidad Politécnica de Atlautla · Grupos 171 y 172

Compara **Dijkstra** (búsqueda de costo uniforme, sin heurística) y **A\*** (heurística de distancia geodésica) para encontrar la ruta más corta en una red carretera regional: Chalco · Amecameca · Ozumba · Atlautla · Morelos (16 nodos, 25 tramos).

## Requisitos

- Python 3.10 o superior
- `pip install matplotlib networkx numpy`

## Cómo ejecutar (paso a paso)

1. Clona el repositorio y entra a la carpeta de la sección:
   ```bash
   git clone <URL-DE-TU-REPOSITORIO>
   cd <repositorio>/seccion5
   ```
2. (Opcional) Crea un entorno virtual: `python -m venv venv` y actívalo.
3. Instala dependencias: `pip install matplotlib networkx numpy`
4. Ejecuta: `python astar_dijkstra.py` (se abre la **simulación animada**; clic = pausar/reanudar)
5. Revisa la carpeta `resultados_s5/`:
   - `simulacion_astar_dijkstra.gif` — la animación guardada
   - `dashboard.png` — mapa de la red, tabla y gráficas comparativas
   - `tabla_comparativa.csv` — datos de los 6 escenarios
   - `metricas.json` — métricas completas (admisibilidad, red grande, A* ponderado)
   - `peso_heuristica.png` — efecto de inflar la heurística

## Simulación

La ventana muestra Dijkstra (izquierda) y A* (derecha) buscando la misma ruta, con colores por estado (blanco = sin explorar, amarillo = frontera, azul claro = expandido, rojo = nodo actual) y un camión que recorre la ruta encontrada.

```bash
python astar_dijkstra.py --origen Amecameca --destino Cuautla   # elegir la ruta
python astar_dijkstra.py --velocidad 2                          # más rápido
python astar_dijkstra.py --sin-ventana                          # solo guardar el GIF (sin pantalla)
```

## Cómo funciona

- **Estado:** localidad actual. **Acción:** tomar un tramo a una localidad vecina. **Costo:** kilómetros por carretera.
- **Dijkstra:** expande por menor `g(n)` (costo acumulado).
- **A\*:** expande por menor `f(n) = g(n) + h(n)`, con `h(n)` = distancia haversine en línea recta a la meta.
- **Peso de cada tramo** = distancia haversine × factor de sinuosidad (1.15–1.45). Como el factor es ≥ 1, `h` nunca sobreestima el costo real, así que es **admisible** (el código lo verifica en los 256 pares nodo-meta y comprueba también la **consistencia**).
- Los resultados se validan contra `networkx.dijkstra_path_length`.

## Datos

Las coordenadas son aproximadas y los pesos son estimados (no son kilometrajes oficiales). Para usar datos reales, sustituye `construir_red()` por una red de OpenStreetMap (`osmnx`) o por una matriz de distancias de una API de mapas; el resto del código no cambia.

## Resultados principales

| Hallazgo | Valor |
|---|---|
| Rutas óptimas iguales en ambos algoritmos | 6 de 6 escenarios |
| Reducción de nodos expandidos con A* (red de 16 nodos) | 9 % – 29 % |
| Reducción en red de ~1 500 nodos (40 pares aleatorios) | 780 → 186 nodos (≈ 76 %) |
| Violaciones de admisibilidad | 0 de 256 |

En la red pequeña A* expande menos nodos, pero tarda un poco más por el costo de calcular la heurística; la ventaja de tiempo aparece al crecer la red.

## Autores

Equipo — Grupo 171 · Docente: Mtro. Javier García Lara

## Declaración de uso de IA

Herramienta: Claude (Anthropic). Uso: apoyo en la estructura del código, generación del dashboard y redacción de documentación. Verificación: el código se ejecutó, los resultados se validaron contra NetworkX y las cifras reportadas provienen de la ejecución.
