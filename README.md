# Gemelo digital simulado — pérdida de frutas y verduras hacia Corabastos

Código del capítulo *"Un gemelo digital simulado de la cadena de frutas y verduras hacia
Corabastos: intervenciones de frío, tiempos y sensores IoT, y contraste con la práctica
europea"*. Reproduce todas las cifras, tablas y figuras citadas en el capítulo a partir de
un modelo Monte Carlo por lotes.

## Funcionalidad

El modelo sigue lotes simulados de frutas y verduras a través de cinco etapas de la cadena
(poscosecha, almacenamiento, transporte, central de abastos, comercio minorista). En cada
etapa calcula la fracción del lote que se pierde según el tiempo de residencia y la
temperatura, usando un coeficiente de deterioro tipo Q10. El modelo se calibra para
reproducir el reparto de pérdidas del DNP (2016) y luego se usa para comparar cuatro
escenarios de intervención (cadena de frío en el transporte, menos tiempo en la central,
sensores con alertas, y la combinación de los tres) frente a una línea base.

Además de los escenarios, el código genera:

- Un análisis de sensibilidad uno a uno (12 supuestos, cada uno en su valor bajo y alto).
- Una variación conjunta de 400 combinaciones aleatorias de esos mismos supuestos.
- Contrafactuales (p. ej. marcar lotes al azar en vez de con sensores) y casos extremos
  (transporte corto y/o ya frío) citados en la sección 5 del capítulo.

## Estructura del proyecto

| Archivo | Contenido |
|---|---|
| `digital_twin_simulation.py` | Script principal: modelo, calibración, escenarios, sensibilidad y figuras. |
| `tests.py` | 20 pruebas de sanidad del modelo (sin pytest, framework propio). |
| `requirements.txt` | Dependencias de Python. |

### Archivos de salida (se generan al correr `digital_twin_simulation.py`)

| Archivo | Contenido |
|---|---|
| `scenario_results.csv` | Pérdida total y por etapa, reducción relativa e intervalos de confianza para cada escenario (S0–S4). Corresponde a la Tabla 3 y la Tabla 4 del capítulo. |
| `sensitivity_by_intervention.csv` | Reducción de S1, S2 y S3 en cada valor bajo/alto de los 12 supuestos variados uno a uno. |
| `sensitivity_S4.csv` | Reducción de S4 (combinado) en cada supuesto, ordenada por el rango de variación. Base de la Figura 3 (tornado). |
| `joint_sensitivity.csv` | Las 400 combinaciones aleatorias de supuestos, con la reducción resultante de S1–S4 en cada una. |
| `calibration_and_assumptions.json` | Resultado de la calibración, todos los supuestos del modelo base, y los contrafactuales/casos extremos de la sección 5. |
| `fig1_calibration.png` | Reparto de la pérdida entre los dos grupos de etapas del DNP vs. el modelo. |
| `fig2_scenarios.png` | Pérdida por etapa en cada escenario. |
| `fig3_sensitivity.png` | Gráfico de tornado de la sensibilidad de S4. |

## Especificaciones y requisitos

- Python 3.9 o superior.
- Dependencias: `numpy`, `pandas`, `scipy`, `matplotlib` (ver `requirements.txt`).
- No requiere conexión a internet ni datos externos: todos los supuestos están en el
  propio script (diccionario `BASE_PARAMETERS` en `digital_twin_simulation.py`).

### Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Reproducir todas las cifras y figuras del capítulo

```bash
python3 digital_twin_simulation.py [output_folder]
```

- `output_folder` es opcional (por defecto usa el directorio actual).
- Tarda aproximadamente 80 segundos.
- Al terminar, imprime en consola la calibración, la tabla de resultados por escenario y
  la tabla de sensibilidad, además de escribir los 8 archivos de salida listados arriba.

### Correr las pruebas de sanidad (20)

```bash
python3 -I tests.py [output_folder]
```

- El flag `-I` (modo aislado) evita que Python lea módulos de la carpeta actual o del
  directorio de inicio que no sean los del propio proyecto.
- `output_folder` es opcional. Si se indica (debe contener las salidas ya generadas
  por `digital_twin_simulation.py`), se agregan pruebas de consistencia entre los
  archivos CSV y JSON generados; si se omite, solo se corren las pruebas del modelo en sí.
- Cada prueba es una función `test_*` independiente; al final se imprime cuántas de las
  20 pasaron y el *traceback* de las que fallen, si las hay.

## Reproducibilidad

Todas las fuentes de aleatoriedad (sorteo de tiempos, temperaturas, detección de
excursiones y falsas alarmas) usan generadores de NumPy con semilla fija, por lo que
correr el script dos veces produce exactamente los mismos resultados.

## Licencia

MIT (ver `LICENSE`).
