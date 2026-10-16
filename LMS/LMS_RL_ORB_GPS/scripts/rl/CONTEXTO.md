# Contexto del proyecto — TFG SLAM + GPS + RL

**Última actualización:** 11 de octubre de 2026 (primera prueba de campo: sensores
completos, el EKF los usó en las cuatro sesiones; la cámara sola falla con lluvia en el
parabrisas; la rotonda la sigue el GPS, no la cámara, §4.21. Antes, la laptop de campo: torch con CUDA 12.6
para la GTX 1050 Ti, §14 y §15; con el teléfono por cable la latencia p95 no cumple en
la laptop, §4.20; antes, 9/10: la IMU y la consulta de hora del teléfono real, §4.20)
**Actualización anterior:** 7 de octubre de 2026 (sesión 4: conexión en vivo con la app, grabaciones en carro,
línea base solo visual, corrección del encadenamiento de poses, máscara en vez de recorte,
causa de los saltos de rumbo, red de seguridad en 10°, verificación en KITTI, EKF partes 1
y 2, diagnóstico del lazo en autopista, prueba por cable, trabajo sin internet, simulador
del teléfono, zona UTM fija en el replay, interfaz con video y su prueba con el
teléfono por cable; decisión: el rumbo sale del giroscopio)
**Dónde quedamos:** §10, apartado "Dónde quedamos". Leerlo primero al retomar.
**Estudiante:** Naheem Johnson Solís — TEC, Escuela de Ingeniería en Computadores
**Documentos hermanos:** `realtime/CONTEXTO_APP_ANDROID.md` (plan original de la app) y
`realtime/ESTADO_APP_ANDROID.md` (lo que efectivamente se construyó, protocolo de red
y observaciones para el pipeline; **leerlo antes de tocar las fuentes en vivo**)
**Propósito:** permitir retomar el trabajo en una sesión nueva sin reconstruir el contexto.

---

## 0. RESUMEN EN 18 LÍNEAS (leer esto primero)

1. TFG del TEC: plataforma SLAM híbrida (visión monocular + GPS + RL) **en tiempo real**. Entrega diciembre 2026.
2. **El anteproyecto declara que la contribución es el TIEMPO REAL**, no el RL. De los 4 objetivos, solo OE3 depende del RL.
3. **El sistema original (`main.py`) NO fue modificado en ninguna sesión.** Todo lo nuevo vive en `realtime/` y `scripts/rl/`.
4. **Hallazgo central resuelto:** la cámara no tenía escala propia, se la pedía al GPS. Los sensores estaban encadenados y no había nada que fusionar.
5. **Solución encontrada:** escala por **modelo de profundidad monocular** (Depth Anything V2 Metric Outdoor). Da metros sin consumir GPS, **pero con un sesgo que depende del encuadre y de la escena**: de −30 % a +59 % en las grabaciones en carro (§4.13). Sirve como sensor de velocidad que el EKF calibra con el GPS, no como escala absoluta.
6. Validado en datos del teléfono: la trayectoria mejora de **24.64 m a 16.89 m**. En KITTI el error de escala va de **10 % en ciudad a 63 % en autopista** — el 11.4 % que se citaba era una sola secuencia (§3.3).
7. **El pipeline de tiempo real está construido y medido:** 23 ms por frame, 45 ms de latencia p95, 30 Hz sostenidos, 1 % de descarte.
8. El RL del proyecto base **no era RL**: su recompensa premiaba imitar reglas heurísticas. Su techo era la heurística, por construcción.
9. **El margen para el RL está medido y acotado** (§3.5.1): elegir la fuente de escala da hasta 64 %, pero **una regla de un umbral sobre la velocidad captura dos tercios**. Eso vale para *elegir* una fuente; con el EKF el RL ajusta covarianzas y **el margen hay que volver a medirlo** (§3.5.1, al final).
10. **El estimador de escala YA ESTÁ INTEGRADO** al pipeline, en un hilo de fondo
    a 3 Hz: cuesta 2.3 ms de latencia y no mueve los 30 Hz. **La app Android está
    terminada y CONECTADA** al pipeline (§4.9). **Por cable USB** (anclaje de red; el
    S23 necesita un chip, con los datos apagados) no pierde frames (0 %, contra 7 % por
    WiFi) y la PC tarda 26–30 ms por frame. **La latencia de captura a resultado (p95
    113 ms el 5/10) está sin confirmar:** depende del desfase entre el reloj del
    teléfono y el de la PC, que cambia de un día a otro (56 ms atrasado el 29/9, al
    menos 98 ms adelantado el 6/10; §4.9). Funciona sin internet y tiene **interfaz**:
    el botón "SLAM GPS con realtime" de `interfaz.py` abre una ventana con el video, en
    vivo o de una grabación, probada con el teléfono por cable el 6/10 (§10).
    **Falta:** integrar el EKF en vivo y el mapa en esa ventana.
11. Bugs reales identificados y **aún sin corregir en `main.py`**: calibración de KITTI usada en video móvil, `Dropout` activo en inferencia, una variable de estado constante.
12. **Con el vehículo DETENIDO la VO producía basura** (2 inliers de 1381 matches)
    y esa basura giraba toda la trayectoria posterior (§4.6). **Causa encontrada y
    corregida en `realtime/` el 2/10/2026 (§4.13):** un empate dentro de
    `recoverPose`. Para el carro quieto, la razón inliers/matches no sirvió (no separa
    al quieto del lento): la compuerta de detenido usa el residuo tras el mejor giro
    puro entre frames a 0.33 s (§4.15).
13. **Regla de trabajo que funcionó:** medir antes de construir, con puertas de decisión numéricas. Los 12 errores propios de análisis de §6 se encontraron midiendo.
14. **El replay alinea por el reloj de arranque desde el 29/9/2026** (la columna Unix
    de la cámara es la llegada a la app, con 27–71 ms de atraso y saltos). Los
    números de referencia cambiaron: ver §4.5 y §13.
15. **El giroscopio del teléfono entra al sistema: da el rumbo del EKF** (decidido
    por el estudiante con el asesor el 7/10/2026, §4.16). Deriva 1.2–1.7°/min contra el
    curso del GPS; la cámara, 19–55°/min. Hasta ahora era solo instrumento de medición
    (§4.10): para la cámara sola sigue sirviendo de referencia, pero dentro del sistema
    fusionado ya no es independiente, y el rumbo fusionado se evalúa con cortes de GPS.
    La app lo graba pero todavía no lo transmite (pedido firme en §10). Hay **dos
    grabaciones en carro** del 30/9/2026, 12 min cada una (§8).
16. **El encadenamiento de poses estaba al revés en TODO el proyecto** (desde el
    sistema original): `recoverPose` da el movimiento de los puntos y se encadenaba
    como si fuera el de la cámara, lo que espeja las curvas. La alineación 3D lo
    disimulaba. **Corregido en `realtime/` el 2/10/2026** (`camera_positions`, §4.11);
    los scripts viejos siguen con el error.
17. **Línea base solo visual en carro (2/10/2026, §4.12 y §4.13).** Los saltos de
    rumbo eran un empate de `recoverPose` y están corregidos: rumbo máximo de 810° a
    91° y de 294° a 24°. Queda la deriva normal de la VO: con la velocidad del GPS,
    la cámara se separa 16–48 % en 4 min solo por el rumbo (el giroscopio, 4 %).
    La escala del modelo de profundidad depende del encuadre (escala de píxeles) y de
    la escena: −30 % a +59 %. `realtime/evaluate.py` mide todo eso.
18. **EKF con el giroscopio y la velocidad como estado (§4.14–4.17):** `realtime/fusion.py`
    y la compuerta de detenido en el estimador de profundidad. El lazo de la cámara sola
    en autopista era de **rumbo** (§4.16); desde el 7/10/2026 el rumbo sale del
    giroscopio y la velocidad es un estado (`[x, y, θ, s, v]`). Le gana al filtro
    anterior en todos los cortes de GPS simulados, de 10 a 120 s: a 30 s, p50 15.7 contra
    31.6 m (14:05) y 27.3 contra 44.4 m (14:18). El sesgo del giroscopio se estimó y no
    ayudó: queda apagado. **Falta integrarlo al pipeline** (§10).

---

## 1. CONTEXTO DEL PROYECTO

**Título:** "Desarrollo de una Plataforma SLAM Híbrida en Tiempo Real Mediante
Fusión de Visión, GPS y Aprendizaje por Refuerzo para Navegación en Entornos Viales"

**Objetivos específicos y entregables** (entregables actualizados al 27/9/2026):

- **OE1** — analizar requerimientos funcionales, computacionales y **temporales** para tiempo real.
  *Entregable:* documento de requerimientos funcionales, computacionales y
  temporales de la plataforma. Definición de las métricas de evaluación que se
  emplearán en la validación experimental.
- **OE2** — diseñar e implementar la plataforma SLAM híbrida (visión + GPS) en tiempo real.
  *Entregable:* documento de arquitectura del sistema, con la descripción de sus
  módulos y de las interfaces entre ellos. Plataforma SLAM híbrida funcional,
  capaz de operar con información visual y datos GPS.
- **OE3** — desarrollar e integrar un mecanismo de **RL** para optimizar la fusión.
  *Entregable:* módulo de aprendizaje por refuerzo funcional e integrado a la
  plataforma. Configuración de entrenamiento del agente, incluyendo la definición
  del estado observado y de la señal de recompensa utilizada.
- **OE4** — evaluar experimentalmente (ATE, RPE, estabilidad, rendimiento computacional).
  *Entregable:* resultados experimentales obtenidos de los recorridos controlados
  en carretera. Informe de evaluación con el análisis comparativo frente al
  enfoque basado únicamente en SLAM visual.

Los documentos entregables se redactan fuera de este task; aquí solo se trabaja
el código. Lo que cada entregable le exige al código:

- **OE1:** las métricas de evaluación tienen que estar implementadas y calcularse
  igual en todas las corridas. Hoy `pipeline.py` mide latencia, Hz y descarte, y
  `run_replay.py` mide la desviación contra el GPS con alineación (tipo ATE, pero
  el GPS del teléfono no es ground truth). En `realtime/` no hay ATE contra ground
  truth ni RPE; existen en el sistema viejo (`utils/metrics/trajectory_metrics.py`).
  Desde el 2/10/2026 `evaluate.py` mide, contra el GPS y el giroscopio del teléfono,
  el ATE con alineación, la deriva desde un inicio común, la escala y el rumbo.
  En vivo, la latencia de captura a resultado incluye el desfase entre los relojes
  del teléfono y de la PC, y ese desfase cambia de un día a otro: el teléfono iba
  56 ms atrasado el 29/9/2026 y al menos 98 ms adelantado el 6/10/2026 (§4.9). **Desde
  el 9/10/2026 el pipeline mide el desfase en cada sesión** con la consulta de hora de
  la app (§4.20). **Por cable USB (anclaje de red, la configuración de campo) cumple:
  captura a resultado 97/107 ms (p50/p95), medido el 11/10/2026.** Por WiFi no cumple
  el p95 (180–224 ms): el exceso es del enlace. **En la laptop de campo (i5-8300H) por
  cable tampoco cumple el p95: 132–138 / 167–189 ms**, por el cómputo por frame (~50
  ms contra 24–27 en la de escritorio), no por el enlace; la frecuencia sí (~19 Hz).
  Decisión del 11/10: la prueba de campo va así (§4.20).
- **OE2:** la plataforma tiene que operar con visión **y** GPS a la vez. Sin el
  EKF este entregable no se cumple.
- **OE3:** el estado observado y la recompensa tienen que quedar definidos
  explícitamente en el código, no implícitos. El entregable pide un módulo
  funcional e integrado; no exige batir un baseline.
- **OE4:** la comparación es **híbrido contra solo SLAM visual**, sobre los mismos
  recorridos. El pipeline tiene que poder correr una misma grabación en los dos
  modos (con fusión y solo visual). `main.py` **no** es el baseline del OE4
  (decidido el 27/9/2026, ver §2 decisión 1).

**Limitaciones declaradas:** solo cámara monocular y GPS; **sin IMU**; sin ROS;
recorridos controlados. **La de la IMU dejó de regir el 7/10/2026:** el estudiante,
con el asesor, decidió usar el giroscopio para el rumbo (§4.16). Lo que cuenta son
los objetivos específicos, cuya redacción no cambia.

**Clave:** el anteproyecto dice que la plataforma ya existe (proyecto CE-1114,
2025) y que **lo que falta es la validación en tiempo real y en recorridos
viales**. Ese es el TFG. El RL es uno de cuatro objetivos.

---

## 2. ESTADO ACTUAL

### Lo que funciona y está medido

| Componente | Estado | Números |
|---|---|---|
| Pipeline de tiempo real | Construido y probado | 23 ms/frame, 45 ms p95, 30 Hz, 1 % descarte |
| Modo replay (estricto y determinista) | Validado | El determinista predice al estricto con ~1 ms |
| Estimador de escala por profundidad | Validado en 7 secuencias | 10 % ciudad / 63 % autopista (§3.3); mejora la trayectoria móvil |
| **Escala INTEGRADA al pipeline** | Medida en los dos modos, n=3 | +2.3 ms de latencia, 30 Hz sostenidos, 83 estimaciones a 3 Hz |
| Odometría visual base | Sana | 100 % de poses estimadas en las 8 secuencias KITTI |
| Pruebas de invariantes | 6/6 pasan | Más auditoría de fugas de ground truth |
| **Cliente en vivo con la app** (`run_live.py`, `session.py`) | Probado con el simulador y con el teléfono, por WiFi y por cable | Por cable: 30 fps, 0 % perdido, 0.2 % descartado, PC p95 29–30 ms; captura a resultado p95 113 ms, sin confirmar por el desfase de relojes; por WiFi 7 %, 5 % y 212 ms (§4.9) |
| **Interfaz** (`live_window.py`, botón en `interfaz.py`) | Probada en Xvfb con el simulador y con grabaciones, y con el teléfono por cable (6/10) | En vivo o replay, video ~13.5 imágenes/s en pantalla con el procesamiento a 30 Hz, una sola ventana; en vivo muestra la latencia en la PC (§4.9, §10) |
| **Sin internet** | Verificado simulando la red caída | El modelo carga en 4 s (antes 96 s); `setup.sh` lo descarga y avisa si la GPU no sirve (§4.9) |
| Replay de grabaciones de la app nueva | Validado contra el giroscopio | Misma precisión de rotación que en vivo: 0.32° contra 0.33° por frame (§4.10) |
| **Encadenamiento de poses** (`camera_positions`) | Corregido el 2/10/2026, verificado con un experimento sintético | Antes espejaba las curvas (§4.11) |
| **Herramienta de evaluación** (`evaluate.py`) | Reproduce el análisis manual de la caminata del 29/9 (§4.10) | Escala contra Doppler, rotación y rumbo contra el giroscopio, forma contra el GPS (§4.12) |
| **Front-end visual corregido** (máscara, votación con todos los puntos, red de seguridad de 10°) | Verificado en 2 fragmentos en carro, la referencia vieja y 7 secuencias KITTI | Sin giros de 180°; en KITTI el trazo sigue al ground truth con 2–14 % (§4.13) |
| Línea base solo visual en carro | Medida y corregida (§4.12, §4.13) | Rumbo máximo 91° y 24° (antes 810° y 294°); se separa 27–41 % de lo recorrido, 16–48 % aun con la velocidad perfecta: es la deriva normal de la cámara sola |
| **EKF con giroscopio y velocidad como estado** (`fusion.py`, `evaluate.py --fusion --sweep`) | Hecho, evaluado y verificado sobre las grabaciones completas (§4.17, §4.18) e integrado al pipeline: replay, en vivo y la ventana (§4.19) | Predicción a 1 s: 1.3 m p50; cortes de 30 s: 15.7 y 27.3 m p50 (filtro anterior: 31.6 y 44.4; GPS a velocidad constante: 53.8 y 79.1) |
| **Compuerta de detenido** (`depth_scale.py`) | Hecha y evaluada en las grabaciones completas (§4.15) | 85–97 % de los altos, 0.6–1.1 % de la marcha; la cámara dice 0.00 m/s detenida (antes 0.44–0.64) |
| **Giroscopio para el rumbo** (decidido el 7/10/2026 con el asesor) | En el EKF desde el 7/10 (§4.17); falta que la app lo transmita | Deriva 1.2–1.7°/min contra el curso del GPS; la cámara, 19–55°/min frame a frame y 4.5–30°/min cada 10 frames (§4.16) |

### Lo que falta

| Pendiente | Prioridad |
|---|---|
| Pedido a la app: transmitir la IMU y la consulta de hora (§10) | Alta — lo pasa el estudiante al task de la app; mientras tanto el EKF en vivo usa el giro de la cámara |
| Acordar con el asesor el alcance del OE3 con los números de §3.5.1 | Alta — define un objetivo completo |
| RPE en `evaluate.py` y definir cómo se mide la "estabilidad" (los pide el OE4) | Media — antes de la evaluación final |
| Fuente KITTI para `run_replay` (hoy solo lee el formato móvil) | Media — útil para el OE4, por el ground truth exacto |
| El mapa en la ventana (ejes en metros, sin fondo de calles), con la trayectoria fusionada | Alta — **siguiente paso** (el EKF ya corre en las sesiones de la ventana) |
| Implementar el RL | OE3, al final |
| Bugs de `main.py` (§5) | No se corrigen: es referencia histórica, no se usa en la evaluación |

### Decisiones de arquitectura tomadas

**1. `realtime/` es el sistema principal de aquí en adelante.** `main.py` queda
**intacto como referencia histórica** del proyecto anterior (CE-1114). No se borra
ni se modifica, pero **no es el baseline del OE4** (decidido el 27/9/2026): el
entregable del OE4 compara el sistema híbrido contra el mismo pipeline en modo
solo visual, sobre las mismas grabaciones.

**2. El módulo de tiempo real NO importa el código viejo — lo reimplementa.**

`realtime/pipeline.py` (`VisualFrontEnd`) crea su propio `cv2.ORB_create(2000)` y
su propio `BFMatcher`, en vez de importar `PoseGraphSLAM`. Los scripts de
`scripts/rl/` sí lo importan, porque estaban **caracterizando el sistema viejo** —
ahí corresponde usar su código. No es una inconsistencia: son propósitos distintos.

*Por qué no se importa:* `PoseGraphSLAM.__init__` arma estado de keyframes que el
pipeline de tiempo real no usa, y trae por defecto la calibración de KITTI
(`fx=718.856, cx=607.19, cy=185.22`) — que es justamente uno de los bugs
detectados (§5.1). Instanciar esa clase solo para extraerle dos objetos de OpenCV
acoplaría el módulo limpio al viejo arrastrando su equipaje.

*El algoritmo es idéntico:* detectar → emparejar (Lowe 0.75) → matriz esencial →
`recoverPose`, con 2000 features y mínimo 15 matches. **Lo que cambia es cómo se
ejecuta**, y eso es deliberado:

| Aspecto | `main.py` (viejo) | `realtime/` (nuevo) |
|---|---|---|
| Keyframes y acumulación de poses | Por lotes, al final | Incremental |
| Salida por consola | Decenas de `print` por frame | Sin `print` en el bucle |
| Instrumentación | Ninguna | Temporizadores entre etapas |
| `R_align` (dirección del GPS) | Sí | **No** — deliberado |
| Escala del GPS | Sí | **No** — deliberado |

**3. Los parámetros de la VO se verifican, no se comparten.**

**Por qué se mantiene T6 aunque `main.py` ya no sea baseline:** garantiza que la
VO de `realtime/` es la misma que la del sistema histórico, así que los
diagnósticos de §3 (hechos sobre el código viejo) siguen aplicando al nuevo. Ya no
protege la comparación del OE4: esa corre dentro del mismo pipeline y comparte la
VO por construcción. No molesta dejarlo; si algún día estorba, se puede retirar.

*Mitigación IMPLEMENTADA:* el invariante **T6** de `sanity_checks.py` compara los
dos sistemas y falla nombrando el parámetro que divergió. Cubre los ocho
atributos que OpenCV expone del detector (features, escala, niveles, umbral de
borde, WTA_K, patchSize, fastThreshold), el ratio de Lowe y el mínimo de matches.

Detalles de implementación que conviene recordar:
- El mínimo de matches se lee del **fuente** de `main.py` con `ast`, sin
  importarlo: importarlo traería torch y el agente RL, y sus imports fallan
  según desde dónde se ejecute (§5.6).
- La referencia histórica que compara T6 es `RL_ORB_SLAM_GPS.process_frame_with_gps`
  (mínimo 15), **no** `PoseGraphSLAM.process_frame` (mínimo 30). Son distintos.
- El `BFMatcher` no expone su `normType` en Python; solo se compara el tipo.
- Se verificó que T6 **falla** al divergir el mínimo de matches, el número de
  features y el ratio de Lowe. Un invariante que nunca falla no vale nada.

**4. `DepthScaleEstimator` acepta `orb=` y `matcher=` como parámetros opcionales**,
para poder recibir los del código viejo en los scripts de análisis y usar los
propios en producción.

---

## 3. HALLAZGOS PRINCIPALES

### 3.1 El acoplamiento de escala (el hallazgo central)

`cv2.recoverPose` devuelve traslación de **norma 1** — propiedad de la geometría
epipolar, no un detalle de implementación. En `main.py:434`:

```python
escala = distancia_real / distancia_slam   # distancia_slam ≈ 1 SIEMPRE
t[:3] *= escala                            # => ‖t‖ ≈ distancia medida por el GPS
```

Y en `main.py:338-344` (`R_align`) la **dirección** también se rota hacia el GPS.

**Consecuencia: magnitud y dirección de cada paso venían del GPS.** La cámara
aportaba poco más que la rotación.

**Evidencia:** el error de la VO sola crece de **41 m → 80 m → 250 m** cuando el
ruido del GPS pasa de σ=3 a σ=10. Un sensor independiente no se degradaría así.

### 3.2 La solución: escala por profundidad monocular

Modelo: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` (~100 MB,
vía `transformers`).

**Mecanismo:** se triangulan las correspondencias con la pose de norma unitaria
(profundidad en "unidades de VO"), el modelo predice esas mismas profundidades en
metros, y la razón da la escala. Como ‖t‖=1, esa razón **es** la distancia
recorrida.

**Descubrimiento crítico — la base de triangulación:** a 30 fps el vehículo
avanza ~14 cm entre frames vecinos, insuficiente para triangular puntos a
10–50 m. Medido sobre datos reales:

| Separación entre frames | Base | Razón estimada/real | Error |
|---|---|---|---|
| 1 frame | 0.14 m | **2.64** | 164 % |
| 10 frames (~0.33 s) | 2.38 m | **1.08** | 31 % |

**El primer intento falló por esto, no por el modelo ni por la grabación.**

### 3.3 Resultados de la escala

**En KITTI (con ground truth), error de escala:**

| Fuente | Error mediano |
|---|---|
| Derivada del GPS a 1 Hz | ~45 % |
| Plano de tierra (implementación simple) | ~50 % |
| **Profundidad monocular** | **11.4 %** (ver abajo) |

**CORRECCIÓN (14/9/2026): ese 11.4 % era UNA secuencia.** Medido sobre las 7
usables, el error mediano por secuencia es:

| Secuencia | v mediana | Error de escala |
|---|---|---|
| drive_0009 | 12.0 m/s | **10.1 %** |
| drive_0001 | 14.2 m/s | 14.9 % |
| drive_0005 | 8.4 m/s | 15.9 % |
| drive_0018 | 9.9 m/s | 20.3 % |
| drive_0013 | 14.2 m/s | 23.3 % |
| drive_0071 | 8.9 m/s | 36.9 % |
| **drive_0042** | **25.1 m/s** | **63.5 %** (sesgo −62.8 %) |

**La relación con la velocidad tiene forma de U, no es monótona.** Medido sobre
todas las secuencias juntas:

| Velocidad | Error mediano | | Velocidad | Error mediano |
|---|---|---|---|---|
| 0–2 m/s | **57.2 %** | | 10–12 m/s | 14.3 % |
| 2–4 m/s | 30.3 % | | 12–14 m/s | 22.5 % |
| 4–6 m/s | 18.6 % | | 14–18 m/s | 53.7 % |
| **6–8 m/s** | **9.8 %** | | 18–22 m/s | 71.0 % |
| **8–10 m/s** | **11.3 %** | | 22–40 m/s | 70.3 % |

**La ventana útil es ~6–12 m/s (21–43 km/h).** Falla por los dos extremos:
- **Lento:** sin base de triangulación. A 0–2 m/s el error mediano es 57 % y el
  p90 llega a 241 %.
- **Rápido:** la escena se aleja del rango útil del modelo (3–60 m) y a 10 Hz el
  vehículo avanza 2.5 m entre frames.

**Por eso la correlación lineal velocidad–error es +0.019: la U se cancela
sola.** Cualquier análisis que use correlación lineal aquí concluye "no hay
relación", y es falso. (La correlación −0.766 de §3.5.1 es otra cosa: velocidad
contra la *ventaja de la profundidad sobre el GPS*, no contra su error.)

**La secuencia más parecida al tráfico de San José es la que peor sale.**
`drive_0071` tiene 84 % de sus frames bajo 4 m/s (pare y siga) y da 36.9 % de
error — la segunda peor de las siete. El problema para recorridos urbanos en
Costa Rica no es la autopista: es el semáforo.

Es el mismo error de método que §6 advierte dos veces: **se declaró éxito con una
sola secuencia.** La cifra honesta para la tesis es "10–24 % en ciudad, se rompe
en autopista", no "11.4 %".

**Trayectorias en KITTI (ATE, alineación rígida sin escala):**

| Secuencia | Ground truth | GPS 1 Hz | Profundidad |
|---|---|---|---|
| drive_0009 | 2.91 | 36.56 | **4.95** (gana) |
| drive_0013 | 2.47 | **4.34** | 13.29 (pierde) |
| drive_0018 | 0.66 | **18.11** | 35.70 (pierde) |
| drive_0001 | 0.59 | 7.85 | 7.99 (empata) |

**No domina consistentemente.** Gana en una, pierde en dos, empata en una.

**En datos reales del teléfono (Costa Rica, 900 frames, 190 m):**

| Fuente de escala | Desviación vs GPS | Distancia total |
|---|---|---|
| **Profundidad (con mediana móvil)** | **16.89 m** | 215 m |
| Derivada del GPS (método actual) | 24.64 m | 174 m |
| GPS (referencia) | — | 190 m |

Velocidad estimada: **6.85 m/s** contra 6.45 m/s real. La comparación está
sesgada a favor del GPS (se mide contra el propio GPS) y aun así la profundidad
gana.

**A verificar (2/10/2026):** las dos tablas de trayectorias de esta sección (ATE en
KITTI y la de datos del teléfono) salieron de `depth_trajectory_test.py`, que
encadena las poses al revés (§4.11) y alinea en 3D. En recorridos casi planos la
alineación 3D disimula el error, así que pueden estar bien, pero no está
comprobado. Los errores de **escala** de esta sección no se ven afectados: no
dependen del encadenamiento.

### 3.4 Por qué el RL anterior no podía funcionar

- **La recompensa nunca miraba la posición real.** `ground_truth=None` fijo en
  `main.py:510`, así que la rama que usaría el error real (`rl_agent.py:206`) es
  **código muerto**. Lo que quedaba premiaba imitar un árbol de decisión
  heurístico. **El techo del agente ERA la heurística.**
- **`visual_conf` era una constante.** `min(matches/50, 1.0)` con mínimo de 190
  matches → vale 1.0 en el 100 % de los frames de las 8 secuencias.
- **Estado sin normalizar:** `[conf∈[0,1], conf∈[0,1], error∈[0,∞) metros]`. El
  error dominaba por 1–2 órdenes de magnitud.
- **Sin exploración.** La salida determinista del softmax se usaba como acción y
  como distribución; nunca se muestreaba. La pérdida resultaba ser auto-imitación.
- `gamma=0.95` declarado y **nunca usado**; `next_state` siempre `None`. No hay MDP.
- El bonus de entropía tiene el **signo invertido** respecto de su intención.

### 3.5 El margen para el RL

Descomposición de varianza del peso óptimo: se separa lo que **se repite entre
realizaciones de ruido** (predecible) de lo que **cambia con cada sorteo** (azar).

| Planteamiento | Margen | Aprendible | Veredicto |
|---|---|---|---|
| Ponderar posiciones (el original) | 17 % | ~25 % | Casi una pared: ~4 % real |
| **Elegir la fuente de escala** | **47 %** | por medir | Mucho más prometedor |

Ese 47 % sale de que GPS y profundidad **ganan en secuencias distintas**:
eligiendo la mejor en cada una, el ATE medio baja de 16.72 m (solo GPS) a
**8.81 m**.

**Es un oráculo por secuencia, con 4 muestras.** Un sistema real decide frame
a frame.

### 3.5.1 LA PUERTA DEL OE3, CERRADA (14/9/2026)

`scripts/scale/scale_decision.py`. Oráculo **por frame** (no por secuencia)
sobre las 7 secuencias, con el GPS degradado sintéticamente y validación
**dejando una secuencia afuera**.

**1. El margen existe** (error medio de escala, m/paso):

| σ del GPS | Solo profundidad | Solo GPS | Oráculo | Margen |
|---|---|---|---|---|
| 1 m | 0.607 | 0.114 | 0.089 | 22.7 % |
| 3 m | 0.607 | 0.224 | 0.114 | 49.3 % |
| 5 m | 0.607 | 0.400 | 0.142 | **64.6 %** |
| 10 m | 0.607 | 1.040 | 0.256 | 57.8 % |

**2. Pero el margen sale de UNA secuencia.** La profundidad gana en 6 de 7; solo
pierde en `drive_0042` (autopista), y ahí pierde por mucho (1.329 contra 0.199).
Quitando esa secuencia el margen cae de 64.6 % a **16.7 %**.

**3. Lo captura una regla de UN SOLO UMBRAL, no un modelo aprendido:**

| σ del GPS | Margen | Regresión con 6 features | Regla de 1 umbral sobre la velocidad |
|---|---|---|---|
| 1 m | 22.7 % | **−1656 %** | −49 % |
| 3 m | 49.3 % | −212 % | **+71.5 %** |
| 5 m | 64.6 % | −91 % | **+66.3 %** |
| 10 m | 57.8 % | −35 % | +6.9 % |

Captura negativa = **peor que no decidir**. Con 7 secuencias de dominios
distintos, 6 features alcanzan para memorizar el dominio de entrenamiento; un
solo parámetro no puede.

**4. La señal dominante es la velocidad** (correlación con la ventaja de la
profundidad): `v_gps` **−0.766**, `n_pts` +0.513, `dispers` −0.243,
`razon_inliers` +0.02 (nada).

**CONSECUENCIA PARA EL OE3:** el baseline a superar **ya no es "la mejor
constante" sino la regla de umbral sobre velocidad**. Un agente de RL tiene que
batir eso, no al hombre de paja. Y el margen que queda sin capturar es
0.229 → 0.142 m, o sea **un tercio del margen**. Ese tercio es el techo real del
OE3, y conviene declararlo así con el asesor (§11.2).

La regla solo funciona en el régimen σ = 3–5 m. A σ = 1 m el GPS es tan bueno
que decidir estorba; a σ = 10 m ya casi no hay qué elegir.

### ESTE NÚMERO NO SE TRASLADA AL EKF — hay que volver a medirlo

El "un tercio" es de un planteamiento de **elección dura**: usar profundidad *o*
usar GPS. Un EKF **mezcla** con pesos continuos, que es otro problema:

1. **Mezclar domina a elegir.** Una combinación ponderada de dos estimaciones
   ruidosas puede ser mejor que ambas; una elección dura nunca. Así que **un EKF
   de covarianzas fijas ya se va a comer parte de ese 64.6 % por sí solo**, sin
   aprender nada.
2. **Cambia el baseline.** El techo del RL no se mide contra "la mejor
   constante" ni contra la regla de umbral, sino contra **un EKF fijo bien
   ajustado** — que es un rival bastante más fuerte. La regla de velocidad no
   desaparece: se vuelve una covarianza `Q` programada por velocidad, y *eso*
   es la vara.
3. **Pero abre una dimensión sin medir.** La medición de §3.5.1 solo miró error
   de **escala**. El EKF también estima **rumbo**, y el problema del vehículo
   detenido (§4.6) es de rumbo. Ahí la covarianza es exactamente la perilla
   correcta y **nunca se midió el margen**. Es el único lugar donde al RL le
   queda un caso genuino.

**Conclusión: rehacer la puerta de §3.5.1 cuando el EKF exista**, con la misma
metodología: EKF fijo bien ajustado como baseline, covarianza-oráculo como
techo, RL en el medio. Medir antes era imposible: no había contra qué.

### 3.6 El pendiente que decide si el RL es viable

**No se ha encontrado una señal observable que diga cuándo cada sensor falla.**
Se probaron dispersión de razones, número de puntos y paralaje mediana: las tres
dieron correlación **~0.00** con el error.

**Ese resultado negativo se midió ANTES de corregir la base de triangulación.
No vale y hay que repetirlo.**

Motivo para el optimismo moderado: los modos de falla tienen causas físicas al
menos parcialmente observables (poco paralaje, poca textura, baja velocidad).

**ACTUALIZACIÓN 8/9/2026: apareció una candidata que sí separa** — la razón
inliers/matches, 0.00 con el vehículo detenido contra 0.25 en marcha. Ver §4.7.
Falta validarla sobre las 8 secuencias de KITTI.

---

## 4. EL PIPELINE DE TIEMPO REAL

### 4.1 Arquitectura

```
[hilo captura] --cola tam.1, descarta lo viejo--> [hilo procesamiento]
[hilo GPS]     --GpsBuffer causal-------------->        |
[hilo profundidad, ~3 Hz] --velocidad (m/s)---->        |  (conectado)
                                                        v
                                            [métricas: latencia, Hz, descartes]
```

**Cómo se usa la velocidad:** el paso métrico de cada frame es
`t * (v · dt)`, con `t` la dirección de norma 1 que da la cámara a 30 Hz, `v`
la última velocidad del modelo y `dt` el intervalo **real** entre frames
procesados (no 1/30 s: con descartes los pasos no son uniformes, y suponerlos
uniformes acortaría la trayectoria justo cuando el sistema está más cargado).

Entre actualizaciones del modelo la velocidad se **sostiene**. Es una
aproximación explícita —velocidad constante durante ~0.33 s— que vale porque la
velocidad de un vehículo cambia despacio; no valdría para la dirección, que por
eso sigue saliendo de la cámara a 30 Hz.

**Regla no negociable:** cola de tamaño 1 que descarta lo viejo. Si se encola
todo, la latencia crece sin límite aunque el throughput se vea bien.

### 4.2 Módulos

| Archivo | Qué hace |
|---|---|
| `realtime/sources.py` | Fuentes intercambiables + `GpsBuffer` con acceso **causal**. `GpsBuffer` guarda la velocidad Doppler y entrega los fixes nuevos en orden (`between`); `ImuBuffer` para las muestras de la IMU. Replay: `ReplayFrameSource` (acepta `skip_start`), `ReplayFeed` (fixes o IMU en tiempo real), `load_mobile_session`, `load_mobile_imu`. En vivo: `read_hello`, `LiveFrameSource` (TCP), `LiveGpsSource` (UDP: `START`/`STOP`, líneas `GPS` e `IMU`). `utm_zone` fija la zona UTM con el primer fix en los dos modos y en `evaluate.py`; `HELLO_S23` es el saludo del teléfono |
| `realtime/pipeline.py` | Hilos, cola con descarte, `VisualFrontEnd`, `DepthScaleWorker`, métricas, `simulate_drops`, y la fusión en cada frame (`fusion=`, §4.19). En vivo mide además la latencia de llegada a resultado (`pc_ms`) |
| `realtime/run_replay.py` | Punto de entrada del replay. Flags `--strict`, `--trajectory`, `--save-frames`, `--scale {none,depth}`, `--scale-hz`, `--mask-bottom`, `--fx`, `--fy`, `--cx`, `--cy`, `--skip-start`, `--start`, `--visual-only`, `--no-gyro`; la fusión corre siempre y usa el `gyro_accel.csv` de la grabación. Contiene también lo que comparte con el modo en vivo: `camera_positions` (encadenamiento correcto, §4.11), `TrackRecorder`, `save_frames_csv`, `umeyama`, `plot_trajectory` |
| `realtime/evaluate.py` | Evalúa un `frames.csv` contra el GPS (posiciones y Doppler) y el giroscopio de la grabación: escala por banda de velocidad, rotación por frame y cada 0.5 s, rumbo, ATE y deriva desde un inicio común (§4.12). Con `--fusion` y `--outage`, la fusión y los cortes de GPS simulados (§4.14); con `--sweep`, las variantes del filtro sobre un corte cada 10 s de 10 a 120 s (§4.17) |
| `realtime/fusion.py` | EKF plano `[x, y, θ, s, v]` (`PlanarEKF`), el giro del giroscopio entre frames (`GyroYaw`), su alimentación en orden temporal (`GpsCameraFusion`, con el modo solo visual y el alcance físico de un fix forzado), la etapa del pipeline (`OnlineFusion`) y `run_fusion` para correrlo sobre una sesión guardada (§4.14, §4.17–4.19) |
| `realtime/session.py` | Una corrida del pipeline, en vivo (`LiveSession`) o de una grabación en tiempo real (`ReplaySession`): conectar, preparar, correr, detener desde otro hilo, estado y guardado; corre la fusión (opciones `visual_only` y `gyro`) y en vivo guarda la IMU recibida como `gyro_accel.csv`. La usan `run_live.py` y la ventana |
| `realtime/live_window.py` | Ventana tkinter "SLAM GPS en tiempo real": fuente (teléfono o grabación), opciones, video con la línea de la máscara, estado y resumen al terminar. Se abre desde `interfaz.py` o sola |
| `realtime/phone_simulator.py` | El teléfono simulado: sirve una grabación con el protocolo de la app (incluidas las líneas `IMU` pedidas; `--no-imu` las deja afuera, como la app actual), para probar el modo en vivo y la ventana sin el celular |
| `realtime/run_live.py` | La consola de `LiveSession`. Flags `--record`, `--duration`, `--scale`, `--mask-bottom` (0.25 por defecto), `--video-port`, `--gps-port`, `--visual-only`, `--no-gyro`. Guarda cada sesión en `resultados/realtime/live_<fecha>/` (`frames.csv` con el estado fusionado, `gps.csv`, `gyro_accel.csv` si llegó la IMU, `metrics.json`, gráfico) |
| `realtime/depth_scale.py` | `DepthScaleEstimator` — construido, validado e **integrado**. Desde el 4/10/2026 reporta 0 m/s con el carro detenido (§4.15); desde el 5/10 carga el modelo sin consultar internet si ya está en el disco y cae a la CPU si la GPU no sirve |

El código de `realtime/` está en inglés (comentarios, identificadores y flags);
la salida por consola sigue en español porque es evidencia para el informe.

`DepthScaleWorker` reutiliza los keypoints y descriptores que el front-end ya
calculó (`VisualFrontEnd.last_features`): ahorra ~9 ms de ORB por estimación y
garantiza que ambos módulos vean las mismas features. Lo que **no** comparte son
los objetos de OpenCV: corren en hilos distintos y cv2 no garantiza que sean
seguros.

### 4.3 Métricas medidas (900 frames de datos móviles)

| | Determinista (predicción) | Estricto (real) |
|---|---|---|
| Descartes | 9 (1.0 %) | 9 (1.0 %) |
| Latencia p50 | 25.0 ms | 26.3 ms |
| Latencia p95 | 43.9 ms | 45.1 ms |

Desglose por frame: ORB 9.0 ms, emparejamiento 5.6 ms, pose 7.4 ms.

**Costo de la escala métrica (medido el 8/9/2026, mismos 900 frames).**
El modo estricto **no es determinista**, así que se corrió 3 veces cada
configuración y se reporta la mediana con el rango observado:

| Modo estricto (n=3) | Sin escala | Con escala (hilo de fondo) |
|---|---|---|
| Latencia e2e p50 | 27.0 ms  [26.9–27.6] | **29.3 ms  [28.2–29.4]** |
| Latencia e2e p95 | 47.3 ms  [47.0–47.9] | **49.9 ms  [49.1–50.3]** |
| Hz sostenidos | 29.9 | **29.9–30.0** |

**El hilo de profundidad cuesta 2.3 ms de latencia p50 y 2.6 ms de p95, y no
mueve los 30 Hz.** Los rangos de las dos configuraciones **no se solapan**, así
que el costo es real y no ruido de corrida a corrida.

Una sola corrida de cada lado había dado "+1.2 ms". Ese número salía de
comparar la peor corrida del control contra la mejor con escala. **En un modo
declarado no determinista, n=1 no alcanza para reportar una diferencia de
milisegundos.**

El modelo tarda 88 ms por estimación cuando corre solo y ~97 ms compitiendo con
el bucle principal (p95: 102 → 127 ms), pero eso no se le traslada al frame:
83 estimaciones a 3 Hz, ninguna descartada por cola llena, 79-80 aceptadas y
2-3 falladas.

**Determinismo verificado:** dos corridas del modo determinista con la escala
activa dan resultados idénticos bit a bit. En ese modo el modelo corre **en
línea** a los mismos instantes de datos, y su costo se mide aparte en vez de
sumarse al presupuesto por frame — que es lo que el modo estricto valida.

### 4.5 Trayectoria con escala propia (900 frames móviles)

> **Desactualizado desde el 2/10/2026 por la tarde:** con la máscara y la corrección
> de `recoverPose` (§4.13) la referencia da 254 m (+34 %), RMSE 17.29 m y mediana
> 17.85 m; sin escala, 21.61/20.93 m. La tabla de abajo es la del recorte.

**Números vigentes hasta el 2/10/2026** (replay alineado por el reloj de arranque,
§0 punto 14, y encadenamiento de poses corregido, §4.11). Modo determinista,
900 frames:

| | Sin escala métrica | Con escala por profundidad |
|---|---|---|
| Alineación | Umeyama **con escala** (le presta la escala al ajuste) | **RÍGIDA** (la escala es de la cámara) |
| Desviación RMSE | 15.49 m | **16.59 m** |
| Desviación mediana | 15.27 m | **11.37 m** |
| Desviación máxima | 27.19 m | 36.09 m |
| Distancia recorrida | — (sin unidades) | 204 m contra 189 m del GPS (**+8 %**) |
| Frames iniciales sin GPS | 27 | 27 |

Historia de estos números: con la alineación por la columna Unix y el
encadenamiento viejo daban 15.52/13.73 m (sin escala) y 15.08/11.89 m con 212 m
(+12 %, con escala); con la alineación nueva y el encadenamiento viejo,
15.34/13.43 m y 16.54/11.39 m. Que el encadenamiento casi no los mueva confirma
que en un tramo casi recto la alineación 3D lo disimulaba.

**Qué cambió con la alineación y qué no.** En modo determinista la VO no usa las
marcas de tiempo: sus rotaciones y direcciones son idénticas antes y después. Lo
que cambió es qué fix se empareja con cada frame y el intervalo `dt` con que la
escala pasa de velocidad a metros. Por eso:
- La distancia mejora (+12 % a +8 %): los `dt` de la columna Unix saltaban.
- Los 27 frames sin GPS son los correctos: el primer fix llega 0.88 s después del
  primer frame. Los 19 de antes eran un efecto de la columna Unix.
- El RMSE con escala sube: lo domina el giro basura del alto inicial (§4.6), y
  pasos de otro largo lo reparten distinto. RMSE y distancia no se mueven juntos.

**Cómo leerlo:** la comparación está sesgada **en contra** de la escala por
profundidad —a la VO libre se le permite ajustar su escala contra el propio GPS
con el que se la compara, y a la métrica no—. Con la alineación vieja la métrica
empataba en RMSE y ganaba en mediana; con la nueva pierde en RMSE y sigue ganando
en mediana.

### 4.6 De dónde sale el error de esa trayectoria (diagnosticado el 8/9/2026)

> Las cifras de esta sección son de antes de la corrección de alineación (§4.5) y
> del encadenamiento (§4.11). La conclusión no cambia: las rotaciones de la VO, que
> son las que fallan en el alto, no dependen de ninguna de las dos cosas, y la
> línea base en carro la confirma con el giroscopio (§4.12).

**El error NO lo produce la escala. Lo produce un alto de 3.9 s al inicio.**

| Régimen | Velocidad de la cámara | Velocidad del GPS |
|---|---|---|
| t = 0–4 s (detenido) | 0.23 m/s | 0.04 m/s |
| t = 4–8 s (acelerando) | 4.95 m/s | 4.99 m/s |
| t = 8–30 s (en marcha) | 7.38 m/s | 7.39 m/s |

**La escala coincide con el GPS dentro del 1 % en los tres regímenes.** Lo que
falla es la *dirección*: durante el alto no hay base de triangulación, la matriz
esencial es degenerada y `recoverPose` devuelve rotaciones arbitrarias. Esas
rotaciones quedan acumuladas en la cadena de poses y **giran todo lo que viene
después**.

Evidencia — recortando el arranque, con el mismo código y los mismos datos:

| Empezar a acumular desde | RMSE contra el GPS | Sinuosidad (camino/neto) |
|---|---|---|
| t = 0 s | 15.08 m | 1.576 |
| t = 4 s | 12.22 m | 1.570 |
| t = 8 s | **8.21 m** | 1.370 |
| t = 10 s | **7.77 m** | 1.257 |

(la sinuosidad del GPS es 1.035: el recorrido es casi recto)

**Atribución escala vs forma:** ajustarle la escala a la trayectoria métrica solo
mejora el RMSE un 11 % (15.08 → 13.36 m, factor 1.146). La VO libre con escala
ajustada da 15.52 m. Las tres versiones rondan los 13-15 m porque comparten la
*forma*. El problema está en la orientación, no en la magnitud.

Queda un **+12 % de distancia total** (212 m contra 189 m) que sí es sesgo de
escala, consistente con el ~11 % medido en KITTI (§3.3). Con la alineación nueva
baja a +8 % (§4.5): parte de ese sesgo eran los saltos de la columna Unix.

### 4.7 Una señal de confianza observable, por fin

**La razón inliers/matches delata el régimen degenerado:**

| | Matches | Inliers | Razón |
|---|---|---|---|
| Detenido (t=0–4 s) | 1381 | **2** | **0.00** |
| Acelerando (t=4–8 s) | 910 | 25 | 0.03 |
| En marcha (t=8–30 s) | 1058 | 268 | **0.25** |

Esto importa para el **OE3**: §3.6 dice que las tres señales probadas
(dispersión de razones, número de puntos, paralaje mediana) dieron correlación
~0.00 con el error. Esta no: separa los regímenes por dos órdenes de magnitud.

**VALIDADO EN KITTI (8/9/2026), con el cache existente.** De las 7 secuencias
usables, 3 contienen ambos regímenes:

| Secuencia | Razón si v<1 m/s | Razón si v>3 m/s | Separación |
|---|---|---|---|
| drive_0009 | 0.003 | 0.699 | 267x |
| drive_0018 | 0.023 | 0.487 | 21x |
| drive_0071 | 0.037 | 0.493 | 13x |
| **mediana** | **0.023** | **0.493** | **21.6x** |

**PERO NO SIRVE CUANDO EL VEHÍCULO SE MUEVE.** Correlación entre la razón de
inliers y el error de rotación por paso, solo en frames con v>3 m/s: +0.48,
+0.22, −0.04, +0.20, +0.30, −0.16, +0.36. Mediana **+0.22**, con el signo al
revés de la hipótesis: es ruido, no señal.

Y el motivo es más interesante que el resultado: **en marcha no hay error que
predecir.** El error de rumbo por paso es de 0.1–0.7° tanto en el cuartil de
razón baja como en el de razón alta. La VO acierta la rotación siempre que haya
movimiento.

**Conclusión: la señal es prácticamente binaria — "hay geometría" o "no la hay".**
Sirve para compuerta, no para ponderar. **En el carro no alcanzó** (4/10/2026, §4.15):
con el capó tapado, a 1–3 m/s da lo mismo que detenido; la compuerta usa otra señal.

**Alcance de esta medición:** se midió error de **rotación**. El planteamiento
del RL con 47 % de margen (§3.5) es sobre **elegir la fuente de escala**, que es
otro error. Eso sigue sin medir, porque de las 8 secuencias solo `drive_0009`
tiene estimaciones de profundidad cacheadas (199 muestras). Medirlo exige correr
el modelo de profundidad sobre KITTI.

**Dato relevante para la tesis: la antigüedad mediana del fix de GPS es ~500 ms
(508 ms con la alineación vieja, 491 ms con la nueva), veinte veces más que el
cómputo.** El cuello de botella es el sensor, no la computadora.

### 4.4 Dos modos de replay

- **Estricto** — duerme y descarta de verdad. Reproduce la calle. **No determinista.**
- **Determinista** — procesa todo, mide, y calcula *después* qué se habría
  descartado. **Reproducible; es el modo para desarrollar y reportar.**

### 4.8 Mapa del recorrido completo (`mobile_data/2025_03_11`, 289 s)

Primer replay de la ruta entera (8673 frames, determinista, escala por
profundidad, 29/9/2026). Gráfico en `resultados/realtime/replay_2025_03_11_completo/`.

- El pipeline procesa todo sin errores: 813 de 822 estimaciones de escala aceptadas.
- Distancia: 1309 m contra 1090 m del GPS (+20 %). RMSE 113 m.
- **La forma no se parece al lazo del GPS: queda arrugada en ~180 m.** La ruta
  tiene cuatro altos de 3–11 s entre los segundos 151 y 204, además del inicial.
  Cada alto gira todo lo que viene después (§4.6) y en 1 km se acumula. No lo
  causa la corrección de alineación: las rotaciones de la VO no usan marcas de
  tiempo.
- El GPS de esa grabación termina a los 239 s; el video dura 289 s.

Es la evidencia más clara de que el EKF necesita la compuerta de detenido.

**Ojo:** este mapa se trazó con el encadenamiento viejo (§4.11), que espeja las
curvas. Parte de lo arrugado puede venir de ahí; no se volvió a generar.

### 4.9 Modo en vivo con el teléfono (29/9/2026)

**Arquitectura.** `run_live.py` usa el mismo pipeline que el replay; solo cambian
las fuentes. Una conexión corta lee el `HELLO` para armar `K`; el video se abre
recién cuando el pipeline empieza a leer, para no recibir frames viejos mientras
carga el modelo. Antes de conectar se precalientan ORB (~200 ms la primera
llamada) y el modelo de profundidad (~0.3 s, varios segundos en frío). En vivo
el pipeline corre siempre como el replay estricto.

**Verificación con un simulador del protocolo** (sirviendo `mobile_data`): los
mismos JPEG dan **el mismo resultado** que el replay (diferencia máxima 0.002°,
redondeo del CSV). Las cuatro formas de terminar (5 s sin frames, Ctrl+C,
`--duration`, teléfono inalcanzable) cierran limpio y detienen la grabación.

**Mediciones con el teléfono real (WiFi de la casa):**

| | Escritorio (quieto) | Caminando (90 s, `--scale depth`) |
|---|---|---|
| Frames recibidos | ~25 fps, 16 % no enviados por el teléfono | 2498; 7.5 % no enviados, 6.9 % descartados por la PC |
| PC (llegada a resultado) p50/p95 | 28 / 40 ms | 25 / 44 ms |
| Captura a resultado, medido | 171 / 298 ms | 155 / 260 ms |
| Captura a resultado, sin el desfase de reloj | 115 / 242 ms | ~99 / ~204 ms |
| Antigüedad del fix p50 | 463 ms | 420 ms |

- **El reloj del teléfono iba 56 ms atrasado respecto del de la PC** (medido por USB
  con `adb`, ±5 ms). El protocolo no trae ninguna hora independiente: la hora de
  cada fix sale del mismo reloj del teléfono (proveedor fusionado), no del
  satélite. En campo no se puede corregir sin un pedido a la app (§10).
- **El objetivo de latencia (p95 < 150 ms) no se cumple por WiFi**: la mediana sí,
  el p95 no. El exceso es del enlace (cámara ~65 ms fijos + picos del WiFi), no de
  la PC. Por cable USB el task de la app midió 30 fps sin pérdidas: en campo
  conviene probar anclaje de red por USB antes que el punto de acceso.
- La IP del teléfono cambia (fue .108 y luego .111): revisarla en la pantalla de
  la app o reservarla en el router.

**Verificación del 5/10/2026, con el código actual** (máscara, compuerta de detenido,
red de seguridad), contra un simulador del protocolo rehecho en la carpeta temporal
(sirve 70 s de 14:18 desde el segundo 380, JPEG calidad 80, GPS por UDP):
- `run_live.py` funciona: 30 fps recibidos, 0 faltantes, 1793 procesados y 7
  descartados (0.4 %), 25 ms por frame (p50), 60 fixes a 1 Hz, 177 estimaciones de
  escala sin cola llena.
- **Equivale al replay, no es idéntico:** sobre 1789 pares de frames iguales, el giro
  difiere 0.14° (p50, máximo 4.9°) por la recompresión JPEG; el error contra el
  giroscopio es el mismo (0.35° contra 0.34° por frame) y la distancia de la cámara
  difiere 2 % (661 contra 676 m).
- **Desactualizado o roto:** (1) el EKF no corre en vivo, solo sobre sesiones
  guardadas; (2) `evaluate.py` no lee la carpeta de una sesión en vivo (busca
  `location.csv`, la sesión guarda `gps.csv`); (3) **sin internet el modelo de
  profundidad tarda 96 s en cargar** (5 reintentos a Hugging Face) contra 4 s con
  `HF_HUB_OFFLINE=1`; (4) **`setup.sh` no instala `transformers`** e instala las dos
  versiones de OpenCV; (5) `run_live.py` tiene la máscara en 0 por defecto (en el
  carro hay que pasar 0.22–0.25); (6) `GpsBuffer` no lleva la velocidad Doppler que
  el EKF necesita en vivo. **Corregidos el 5/10/2026 (2) a (5)**, aprobados por el
  estudiante: `evaluate.py` lee `gps.csv`; el estimador de profundidad pasa a
  `HF_HUB_OFFLINE` si el modelo está en el disco (4.1 s sin internet) y cae a la CPU
  si la GPU no sirve con el `torch` instalado (aquí 615 ms por inferencia en CPU
  contra ~100 ms en GPU); `setup.sh` instala `transformers`, una sola versión de
  OpenCV, descarga el modelo y avisa si la GPU no sirve; la máscara de `run_live.py`
  vale 0.25 por defecto. Quedan (1) y (6), que son la integración del EKF.
- **Prueba por cable del estudiante (5/10/2026, escritorio, teléfono en la mano bajo
  techo), contra WiFi el mismo día** (`resultados/realtime/live_20261005_213917/` por
  cable, `live_20261005_211114/` por WiFi):

  | | WiFi (138 s) | Cable USB (124 s) |
  |---|---|---|
  | No enviados por el teléfono | 7.3 % | 0.0 % |
  | Descartados en la PC | 5.2 % | 0.2 % |
  | Captura → llegada p50 / p95 / p99 (máx.) | 88 / 176 / 217 ms (381) | 80 / 85 / 88 ms (113) |
  | Captura → resultado p50 / p95 / p99 | 116 / 212 / 253 ms | 104 / 113 / 120 ms |
  | PC, llegada → resultado p50 / p95 | 27 / 46 ms | 25 / 29 ms |

  **El cable elimina las pérdidas.** Con el desfase de relojes de ese día el p95 dio
  113 ms (< 150 ms), pero queda sin confirmar: el 6/10 se vio que el desfase cambia de
  un día a otro (punto siguiente).
  Por WiFi los frames llegan en ráfagas después de cada hueco y la cola de un lugar
  descarta los viejos; por cable llegan parejos y casi no hay descarte. Los tiempos
  incluyen el desfase de reloj entre teléfono y PC (no se midió; el 29/9 era 56 ms),
  pero el ancho p95 − p50 no depende de él: 5 ms por cable contra 88 ms por WiFi.
  **Para activar el anclaje el S23 necesita un chip**, aunque sea con los datos
  apagados (el estudiante usó el de otro teléfono). La IP del cable fue
  10.145.203.107. En la sesión, 21 poses se descartaron por la red de 10° (giros de
  la mano, más rápidos que los de un carro) y la compuerta marcó detenido el 61 % de
  las estimaciones de escala (la mano casi solo giraba): ambos esperables.
- **Prueba de la ventana con el teléfono por cable (6/10/2026, cuarto, teléfono en la
  mano, sin grabar en el teléfono)** (`resultados/realtime/live_20261006_205834/`,
  76 s): el video se vio en la ventana igual que en el teléfono. 2285 frames
  recibidos, 0 no enviados, 0 JPEG dañados; 2281 procesados y 4 descartados (0.2 %,
  sueltos: 2 justo después del primer frame, que tardó 122 ms, uno a los 6 s y otro a
  los 37 s); 29.9 Hz; PC p50/p95 26/30 ms; 80 fixes (huecos de hasta 5.7 s bajo
  techo, velocidad máxima 0.15 m/s); la compuerta marcó detenido 146 de 229
  estimaciones. Los primeros 5 s, sin puntos ORB (imagen sin textura). Tres hallazgos:
  1. **Captura → llegada dio −98 ms** (p50; p95 −90): ese día el reloj del teléfono
     iba al menos 98 ms adelantado (la PC estaba sincronizada por NTP, −0.9 ms). Con
     el mismo cable, el 5/10 dio +80 ms: el enlace no cambió (ancho p95 − p50 de 7
     contra 6 ms, estable toda la sesión), cambió el desfase entre relojes. Por eso el
     p95 de 113 ms del 5/10 queda sin confirmar, y **en vivo la ventana muestra la
     latencia en la PC** (llegada → resultado); en el replay muestra la de siempre,
     que usa un solo reloj. `metrics.json` guarda las dos.
  2. **Bug de la ventana, corregido el 6/10:** sin "Grabar también en el teléfono",
     el resumen final fallaba (`ValueError: no path specified`: la carpeta del
     teléfono era `None`) y la revisión de estado se cortaba antes de rehabilitar los
     botones, que quedaban todos deshabilitados (la ventana seguía respondiendo y se
     cerraba con la X). No afectó los datos: falló después de guardar. Ahora el estado
     se reprograma al principio y los botones vuelven antes del resumen. Probado en
     Xvfb con cuatro sesiones seguidas en la misma ventana: en vivo con y sin grabar,
     replay, y con el resumen forzado a fallar.
  3. **De Iniciar al primer frame pasaron 34 s**, casi todo cargando el modelo (el GPS
     respondió en 1 s); con la caché caliente son 3.8 s. Probablemente fue la primera
     carga desde que se encendió la PC: torch y CUDA (5.8 GB) están en el disco
     mecánico (`sda`).
- **Simulador del teléfono en el repositorio** (`realtime/phone_simulator.py`,
  aprobado): sirve una grabación con el protocolo exacto, con la cámara en tiempo
  real desde la primera conexión, descarte de frames como la app y la hora Unix
  actual (la latencia medida es la de la PC). Con él, `run_live.py` da 30 fps, 0
  descartes y 29–32 ms de captura a resultado (p50).
- **Cable:** el anclaje de red por USB sirve con la app actual: los puertos escuchan
  en todas las interfaces y la pantalla de la app muestra la IP del anclaje. La
  medición de la app por cable (30 fps sin pérdidas) se hizo con `adb forward`, que
  solo reenvía TCP: sin el GPS por UDP. Falta probar si el teléfono permite el
  anclaje sin chip ni datos.

### 4.10 Validación con el giroscopio y prueba caminando (29/9/2026)

**El método.** La app graba `gyro_accel.csv` (53 Hz) con el mismo reloj que los
frames. Integrando el giroscopio entre dos frames se obtiene la rotación real del
teléfono y se compara con la de la VO. La orientación IMU → cámara se ajustó de
los datos y salió una permutación de ejes con signo, a 1° del ajuste libre:
`[[0,-1,0],[-1,0,0],[0,0,-1]]` (teléfono horizontal, cámara a la izquierda), sin
desfase de tiempo. **Hay que ajustar sobre ventanas de 0.5 s**: frame a frame el
ruido de la VO es del tamaño de la rotación real y el ajuste falla (§6).

**Resultados caminando bajo techo (90 s, teléfono en la mano):**

| Qué | VO | Referencia |
|---|---|---|
| Error de rotación por frame | 0.33° p50, 7.4° p99 | rotación real 0.74° p50 |
| Rumbo a los 90 s | deriva de −119°, hasta 286° | giroscopio |
| Altura | entre −15 y +8 m (con el encadenamiento corregido; el "sube 24 m" que se citaba usaba el viejo) | piso plano |
| Distancia | 128 m | 30–65 m según los pasos del acelerómetro |
| Velocidad estando quieto (45 % del tiempo) | ~1 m/s | 0 |

- **La escala falla quieto y caminando.** Quieto no hay base de triangulación y el
  estimador igual reporta ~1 m/s: la compuerta de detenido tiene que frenar
  también la escala. Caminando duplica la velocidad; hipótesis sin medir: el
  modelo es el de **exteriores** y bajo techo sobrestima.
- **Caminando, la razón inliers/matches no indica calidad de rotación:** los frames
  con razón baja tienen el error más bajo (0.29°). Refleja cuánto se movió la
  cámara. Los umbrales de detenido hay que medirlos en el carro.
- **La dirección de avance frame a frame es ruidosa:** recomprimir a JPEG la cambia
  13° de mediana en el carro (`mobile_data`) y 38° caminando, mientras la rotación
  cambia 0.2–0.3°. El EKF tiene que darle poco peso paso a paso.
- **Replay contra en vivo, mismas 2114 parejas de frames:** igual error contra el
  giroscopio (0.32° y 0.33°). La compresión JPEG de la transmisión no empeora la
  rotación, y el replay de grabaciones de la app nueva es un sustituto válido del
  modo en vivo (lo que necesita el OE4).

Los scripts de este análisis vivían en una carpeta temporal y ya no existen. Si
se usa en las grabaciones en carro, conviene rehacerlo como herramienta del repo.

### 4.11 El encadenamiento de poses estaba al revés (corregido el 2/10/2026)

- `cv2.recoverPose(E, p_anterior, p_actual)` devuelve `R, t` con
  `x_actual = R·x_anterior + t`: transforma **puntos** del frame anterior al
  actual. La cámara se mueve con el inverso: giro `R^T`, desplazamiento `−R^T·t`.
- Todo el proyecto encadenaba `P = P @ [R | t]`: `main.py`,
  `run_replay.plot_trajectory`, `scripts/rl/realistic_rates.py`,
  `scripts/rl/sanity_checks.py` (T4) y `scripts/scale/depth_trajectory_test.py`.
  Eso espeja cada giro y recorre la ruta hacia atrás.
- **Experimento sintético** (cámara que avanza 1 m y gira 10°): `R` coincide con
  `Rc^T` (0.00°); con `P @ [R|t]` la cámara termina en z = −0.985 m, con
  `P @ inv([R|t])` en z = +1 m.
- La alineación 3D de los gráficos lo disimulaba: un recorrido plano espejado se
  arregla girándolo 180° alrededor de un eje horizontal (queda "boca abajo"). Por
  eso en tramos casi rectos los números casi no cambian (§4.5). En el plano
  horizontal, contra el giroscopio o dentro del EKF, el error sí importa.
- **Corregido en `realtime/`:** `run_replay.camera_positions` encadena el inverso y
  la usan el gráfico del replay, el modo en vivo y `evaluate.py`. Los `frames.csv`
  siguen guardando los `R, t` crudos de `recoverPose` (convención de puntos).
- **Sin corregir:** `main.py` y los scripts de análisis nombrados arriba. T4 sigue
  pasando porque propaga solo 1 s con una rotación inicial ajustada. Las tablas de
  trayectorias de §3.3 quedan por verificar.
- **El EKF tiene que usar la convención correcta.**

### 4.12 Línea base solo visual en carro (2/10/2026)

**Cómo se corrió.** Replay determinista con escala por profundidad, intrínsecos del
`HELLO` y `--save-frames`; después `evaluate.py` contra el GPS (posiciones y
velocidad Doppler) y el giroscopio. Resultados en `resultados/realtime/base_14_05/`
y `base_14_18/` (`frames_deterministic_metric.csv`, `trajectory_*.png`,
`evaluation.png`, `evaluation.json`).

| Fragmento | Opciones (además de los intrínsecos y `--scale depth --trajectory --save-frames`) |
|---|---|
| 14:05, 0:00–4:20 (7800 frames) | `--crop-bottom 0.25 --skip-start 6 --frames 7800` |
| 14:18, 6:20–10:20 (7200 frames) | `--crop-bottom 0.22 --skip-start 6 --start 380 --frames 7200` |

(`--crop-bottom` ya no existe: hoy es `--mask-bottom`, con los mismos valores; §4.13.)

Los recortes salen de medir dónde empieza el capó, que refleja la escena: 25 % y
22 % inferior (el tablero empieza en el 15 %). Se promediaron los bordes
horizontales de 80 frames: los de la escena se cancelan y quedan los fijos.

| | 14:05 | 14:18 |
|---|---|---|
| Distancia cámara / GPS | 1455 / 2087 m (−30 %) | 1357 / 1330 m (+1 %) |
| Velocidad cámara/GPS por banda | 0.65–0.77 entre 4 y 14 m/s | 1.16 a 2–6 m/s; 0.87–1.03 a 6–10 m/s |
| Velocidad con el carro quieto | 0.29 m/s | 0.65 m/s |
| Error de rotación por frame p50 / p99 | 0.19° / 1.7° (real 0.20°) | 0.37° / 2.7° (real 0.36°) |
| Rumbo (cámara − giroscopio) | −675° al final, máximo 810° | +72° al final, máximo 294° |
| Separación desde un inicio común | 50 m a 30 s, 983 m a 120 s; final 65 % de lo recorrido | 39 m a 30 s, 247 m a 120 s; final 55 % |
| Altura de la cámara | hasta +460 m | hasta −179 m |

**Causas identificadas:**
1. **Saltos de rumbo.** El error no crece de a poco: salta 80–500° en instantes.
   El giroscopio nunca pasa de 3–6° por paso; los pasos de la cámara con más de 5°
   son el 0.4 % y explican el 56–71 % del error de rotación. **Mecanismo encontrado
   y corregido en §4.13** (un empate dentro de `recoverPose`). Se dijo "sobre todo en
   altos y a baja velocidad": era falso (§6, error 10).
2. **El recorte le baja la escala al modelo de profundidad.** Mismo tramo de 14:05
   (25–95 s, ~10 m/s, 209 estimaciones): cámara/GPS **0.62** con la imagen
   recortada y **0.92** con la imagen completa y los puntos limitados a la zona
   sobre el capó. Se verificó que el mapa de profundidad tiene el tamaño de la
   imagen (no es un error de indexado). Se dijo que en 14:18 el recorte no sesgaba
   la escala; era una inferencia, no una medición: con la imagen completa 14:18 da
   +59 % (§4.13).
3. **Velocidad falsa con el carro quieto** (0.3–0.65 m/s): sin base de triangulación.

**Qué se arregla sin EKF** (simulado sobre los resultados guardados, sin tocar el código):

| | Rumbo máx. 14:05 | Separación final 14:05 | Rumbo máx. 14:18 | Separación final 14:18 |
|---|---|---|---|---|
| Tal como salió | 810° | 65 % | 294° | 55 % |
| Giros de más de 5° por paso tratados como tramo recto | 62° | 39 % | 69° | 34 % |
| Además, cámara quieta en los altos (altos según el GPS: techo de la compuerta) | 50° | 35 % | 68° | 34 % |

Aun así la cámara sola se separa un tercio de lo recorrido en 4 min: la VO
monocular cuadro a cuadro siempre deriva. **El mapa final tiene que ser el
fusionado.** La línea base del OE4 hay que volver a medirla con la profundidad
sobre la imagen completa.

**Lo que esta sección dejaba sin explicar** (el mecanismo de los saltos, la deriva
de altura y el −8 % de escala con la imagen completa) quedó resuelto o medido en
§4.13.

**Dato corregido:** se dijo que en 14:18 "en marcha el rumbo se desvía solo −2°";
salía de saltos que se cancelaban entre sí. Con los saltos corregidos de verdad
(§4.13), 14:18 termina en +1° (máximo 24°) y 14:05 en +91°.

### 4.13 Corrección del trazo: máscara y empate de `recoverPose` (2/10/2026)

**El mecanismo de los saltos.** `cv2.recoverPose` elige entre las cuatro soluciones
de la matriz esencial contando los puntos que quedan delante de las dos cámaras,
pero antes descarta los que están a más de 50 veces el desplazamiento entre frames
(`distanceThresh`, 50 por defecto). Si el desplazamiento es chico comparado con la
profundidad de la escena (detenido, lento o escena lejana), no queda ningún punto:
las cuatro empatan en 0 y OpenCV devuelve la primera. Medido sobre la línea base:
- 52 de los 57 pasos de más de 5° tienen 0 inliers y giran 180°: son la solución
  correcta girada alrededor de la dirección del desplazamiento (coinciden a 0.000°).
- Sin el umbral (`distanceThresh=1e7`), los 111 pasos empatados dan menos de 2° y
  120 pasos normales de control dan el mismo giro. El costo no cambia (5.8 ms).
- Es también la basura del alto inicial de `mobile_data/2025_03_11` (§4.6): en los
  primeros 4 s el código viejo acumula 2700–3000° de giro y el nuevo 22°.
- No son frames repetidos: en todos los empates la imagen cambia respecto de la
  anterior.

**La corrección** (`VisualFrontEnd._recover_pose`): votan todos los puntos. El
conteo de inliers conserva el significado de antes (puntos a menos de 50
desplazamientos, calculado de los puntos triangulados que devuelve `recoverPose`).
Sale idéntico donde la solución elegida coincide y difiere solo donde cambió el
sentido del avance (125 de 483 pasos en un tramo con un alto). La razón
inliers/matches de §4.7 conserva su significado (en el carro, con la máscara, no
alcanzó para detectar el carro quieto: §4.15).

**Red de seguridad:** `VisualFrontEnd.MAX_ROTATION_DEG`. Un paso que gira más se
descarta como "sin pose" (R = I, t = 0), igual que con pocos matches. Sale en el
informe como `poses_giro_imposible`. Lo que separa (escaneado sin compuerta el
3/10/2026):

| Caso | Giro de la cámara | Giroscopio | Matches |
|---|---|---|---|
| Bache a 71 km/h (14:18 completo, 1:13) | 9.7° | 9.5° | 452 |
| Bache de 14:05 (segundo 17) | 5.1° y 6.3° | 6.1° y 5.0° | ~800 |
| Arranque de 14:05, teléfono en la mano (3 pasos) | 11.2°, 14.7°, 15.4° | 1.2–1.6° | 16–46 |
| Alto inicial de la referencia vieja (frames idénticos) | 180° | — | 2000, 0 inliers |

Con 10° se rechazan los cuatro fallos y pasan los dos baches, pero el bache de
9.7° queda a 0.3° del umbral. Con 20° pasan además los tres fallos del arranque.
El 3/10/2026 se subió a 20° con una premisa equivocada (§6, error 12) y ese mismo
día **se volvió a 10°**, decidido con el estudiante: rechazar un bache real cuesta
poco (un frame sin pose, casi todo cabeceo, 1–3° de rumbo) y dejar pasar un fallo
de 15° le mete al EKF un error de rumbo de ese tamaño. Es un límite físico de un
carro, no un ajuste a estas grabaciones.

**La máscara** (`VisualFrontEnd(mask_bottom=...)`, `--mask-bottom` en `run_replay.py`
y `run_live.py`): ORB no busca puntos en la fracción inferior, con un margen de 31
px (el mismo borde que ORB respeta en una imagen recortada), y la imagen ya no se
recorta. Los puntos son idénticos a los del recorte en el nivel fino de la
pirámide; en los niveles gruesos cambia una parte, porque ORB redondea el tamaño de
cada nivel según el alto de la imagen. La rotación contra el giroscopio no cambió
(0.19° y 0.37° p50).

**Por qué el encuadre cambia la escala del modelo de profundidad.** El procesador
del modelo lleva el lado corto de cada imagen a 518 px. Recortar el capó cambia el
lado corto (de 720 a 540 o 561 filas) y con eso la escala de píxeles que ve el
modelo, que no conoce la distancia focal. Profundidad sobre los mismos puntos (70
frames por grabación), relativa a la imagen completa:

| Imagen que recibe el modelo | 14:05 | 14:18 |
|---|---|---|
| Recorte | 0.71 | 0.67 |
| Recorte con una franja negra o gris que devuelve el alto a 720 | 1.00 | 0.99–1.02 |
| Completa, con el capó en negro | 1.00 | 1.02 |
| Esperado si solo importara la escala de píxeles | 0.75 | 0.78 |

En el carro el contenido del capó no importa; importa la escala de píxeles. En el
video vertical de `mobile_data/2025_03_11` el recorte no cambia el lado corto (el
ancho) y aun así da 0.83: ahí sí pesa el contenido (el tablero).

**Resultados.** Replay determinista, mismos fragmentos y opciones que §4.12 con
`--mask-bottom`. Resultados en `resultados/realtime/mask_14_05/`, `mask_14_18/`
(máscara), `mask_vote_14_05/`, `mask_vote_14_18/` (máscara + corrección) y las
figuras comparativas en `comparacion_2026_10_02/`.

| Fragmento | Estado | Distancia cámara/GPS | Rumbo final / máx | Separación 30/60/120 s | Separación final | ATE RMSE |
|---|---|---|---|---|---|---|
| 14:05 | Base (recorte) | −30 % | −675° / 810° | 50 / 346 / 983 m | 65 % | 503 m |
| 14:05 | Máscara | +10 % | +102° / 312° | 51 / 392 / 1196 m | 71 % | 477 m |
| 14:05 | Máscara + corrección | +10 % | +91° / 91° | 111 / 113 / 397 m | 41 % | 98 m |
| 14:05 | Giro de la cámara + velocidad del GPS | — | — | 24 / 91 / 448 m | 48 % | — |
| 14:05 | Giroscopio + velocidad del GPS (control) | — | — | 5 / 15 / 41 m | 4 % | — |
| 14:18 | Base (recorte) | +2 % | +72° / 294° | 39 / 80 / 247 m | 55 % | 262 m |
| 14:18 | Máscara | +59 % | +507° / 725° | 80 / 157 / 192 m | 28 % | 145 m |
| 14:18 | Máscara + corrección | +59 % | +1° / 24° | 83 / 162 / 319 m | 27 % | 120 m |
| 14:18 | Giro de la cámara + velocidad del GPS | — | — | 12 / 70 / 170 m | 16 % | — |
| 14:18 | Giroscopio + velocidad del GPS (control) | — | — | 14 / 20 / 33 m | 4 % | — |

Las dos filas de diagnóstico integran en el plano un giro horizontal con la
velocidad Doppler, como la predicción del EKF con la escala perfecta. La del
giroscopio valida el anclaje: con el rumbo casi perfecto, el método da 4 %.
Velocidad cámara/GPS por banda con la máscara: 14:05, 0.99–1.22 entre 2 y 14 m/s;
14:18, 1.55–1.80. Con el carro quieto, la cámara dice 0.63 y 0.93 m/s.

**Queda explicado:** el mecanismo de los saltos (corregido) y por qué el encuadre
cambia la escala.

**Queda sin explicar o pendiente:**
- **El sesgo de escala por grabación.** Con el mismo encuadre, 14:18 da 1.5 veces
  lo de 14:05; con la máscara, la referencia vieja pasó de +8 % a +34 %. Hipótesis
  sin medir: en 14:05 hay más tránsito, y los vehículos de adelante, que se mueven
  con la cámara, parecen lejanos y bajan la razón metros/unidades. Pista sin
  verificar: si el modelo asumiera la focal de Virtual KITTI 2 (~1000 px a su
  entrada contra 624 px de este teléfono), se esperaría ×1.6, justo lo de 14:18.
- **La deriva continua de rumbo de 14:05** (~20°/min, +76° en marcha). Quedan 12
  pasos de 3–5° sin empate que el giroscopio no ve (alrededor del segundo 150 y
  otros sueltos). Candidatas sin medir: distorsión del lente, obturador rodante,
  objetos en movimiento.
- **El cabeceo:** la cámara deriva 33–42° contra el giroscopio, de ahí las alturas
  de +115 y −543 m. No afecta al EKF plano, que usa solo el giro horizontal.
- **La velocidad falsa con el carro quieto:** le toca a la compuerta de detenido.

**Conclusión:** el trazo solo visual ya no se enreda y sigue al GPS en los primeros
cientos de metros, pero en minutos se separa por el rumbo y por la escala. Con la
escala perfecta, el rumbo solo ya separa 16–48 % en 4 min. La cámara sirve entre
un fix y el siguiente (1 s); el mapa que se muestra tiene que ser el fusionado, con
`s` calibrada por el Doppler.

**Referencia de 900 frames** (`mobile_data/2025_03_11`; resultados en
`resultados/realtime/ref_900_mask/` y `ref_900_mask_vote/`): con la máscara y la
corrección, 254 m contra 189 m del GPS (+34 %), RMSE 17.29 m y mediana 17.85 m; sin
escala, RMSE 21.61 m y mediana 20.93 m. El RMSE con escala lo domina el +34 % (un
tramo recto estirado así da ~18 m por sí solo); sin escala, los frames del alto
inicial cuentan como pasos de largo 1 y deforman el principio. Ya no sirve como
número de regresión de forma; sigue sirviendo para los tiempos (23.9 ms p50).

**Verificación en KITTI** (otra cámara, otro carro, ground truth exacto). Las 7
secuencias usables, con la distancia real de cada paso para comparar solo la forma
(script temporal; figura en `resultados/realtime/comparacion_2026_10_02/kitti_trazos.png`).
Separación final como % de lo recorrido:

| Secuencia | Distancia | Empates | Código original | Solo encadenamiento | Las dos correcciones |
|---|---|---|---|---|---|
| 0001 | 107 m | 0 | 27 % | 2 % | 2 % |
| 0005 | 69 m | 0 | 31 % | 13 % | 11 % |
| 0009 | 332 m | 18 (9 de 180°) | 75 % | 4 % | 4 % |
| 0013 | 173 m | 0 | 5 % | 5 % | 5 % |
| 0018 (casi todo detenido) | 51 m | 22 (14 de 180°) | 73 % | 52 % | 7 % |
| 0071 (lento) | 238 m | 3 | 9 % | 9 % | 9 % |
| 0042 (autopista) | 2590 m | 0 | 140 % | 16 % | 14 % |

Las correcciones son generales: en KITTI el trazo sigue al ground truth con 2–14 %.
El teléfono en el tablero da 16–48 % en la misma prueba (con el Doppler, §4.13):
la cámara de KITTI va en el techo, con obturador global, imágenes rectificadas y 10
fps (más paralaje por paso). Con la escala del modelo de profundidad se suma su
error (en KITTI, 10 % en ciudad y 63 % en autopista, §3.3).

### 4.14 EKF, primera parte: el núcleo `[x, y, θ, s]` (4/10/2026)

**Dónde está:** `realtime/fusion.py` (`PlanarEKF`, `GpsCameraFusion`, `run_fusion`)
y `evaluate.py --fusion --outage INICIO DURACION` (repetible). Corre sobre los
`frames.csv` guardados; **todavía no está en el pipeline** (ni en vivo ni en el
replay). Resultados en `resultados/realtime/fusion_14_05/` y `fusion_14_18/`
(`fusion.png`, `evaluation.json`, `evaluate.log`), con cortes en 60 s (10 s) y
120 s (30 s), elegidos de antemano.

**Diseño:**
- **Predicción** en cada frame: rumbo += giro de la cámara alrededor de la
  vertical; posición += s × paso del modelo de profundidad, en la dirección del
  rumbo. El carro avanza en la dirección de su rumbo (sin desplazamiento lateral
  ni reversa): **no se usa la dirección de la traslación de la VO** (13° de ruido
  por frame, §4.10). Un frame sin pose avanza igual, con giro 0 e incertidumbre
  de 2°.
- **La vertical sale del montaje** (eje y de la cámara). Medido con el acelerómetro
  como instrumento: la gravedad queda a ~3° del eje y (cabeceo +2°, alabeo −2 a −3°)
  y el rumbo con el montaje es igual al rumbo con el acelerómetro (contra el
  giroscopio, final −3.8° contra +0.7° en 14:18 y +87° contra +91° en 14:05).
  **El paso "vertical sin IMU" del plan queda resuelto con el montaje**, salvo que
  el teléfono se monte inclinado.
- **Correcciones** con cada fix: posición (σ 4 m), velocidad Doppler = s ×
  velocidad de la cámara, y **curso entre fixes consecutivos** (la cuerda de 1 s
  es el rumbo a mitad del intervalo; se le suma el giro de la cámara desde
  entonces). Desviación del plan, que decía "rumbo solo por la covarianza cruzada":
  el GPS del teléfono es muy suave entre fixes (sus pasos coinciden con el Doppler
  con 0.3 m de MAD), y el curso baja la incertidumbre del rumbo de 5.1° a 3.6° en
  14:05 con mejoras chicas en los cortes. Si la app manda `getBearing`, se
  reemplaza por esa medida.
- **Arranque:** cuando el GPS ve el carro avanzar 20 m a ≥3 m/s (el rumbo inicial
  es el curso de esa cuerda más el giro desde su mitad), con s = 1 ± 0.5.
- **Compuerta de innovación** al 99.9 %, con aceptación forzada después de dos
  rechazos seguidos (se infla la covarianza hasta que la medida entra).
- La velocidad Doppler no corrige s por debajo de 2 m/s (la razón no significa nada).

**Ruidos, medidos sobre las dos grabaciones antes de correr el filtro:**

| Parámetro | Valor | Medición |
|---|---|---|
| Rumbo de la cámara | 2.2°/√s | contra el giroscopio: 2° en 1 s, 5° en 5 s, 8° en 10 s |
| Velocidad de la cámara | ruido 20 %; s con deriva de 10 %/√s | razón Doppler/cámara: ruido blanco 12–19 %, deriva 9–13 %/√s (función de estructura) |
| GPS | 4 m posición, 0.3 m/s Doppler | entre 3 y 6 m los resultados casi no cambian |

**Dos fallos encontrados durante el desarrollo:**
1. Con la compuerta sola, en 14:05 a los 123 s (acelera a 17 m/s y la escala del
   modelo cambia de golpe) el filtro rechazó la velocidad porque s venía atrasada,
   después la posición, 7 fixes seguidos, y quedó a 41 m del GPS. Se resolvió con la
   aceptación forzada y con la deriva de s medida (había supuesto 5 %/√s).
2. Compensar el atraso de ~1 s de la velocidad de la cámara (la mediana móvil del
   estimador; la correlación con el Doppler es máxima a 1.0–1.3 s) **empeoraba** la
   predicción a 1 s: 2.2 contra 1.8 m en 14:05 y 2.1 contra 1.4 m en 14:18.
   Comparada en el mismo instante, s absorbe el atraso, que es lo que necesita la
   predicción. Se descartó.

**Resultados:**

| | 14:05 | 14:18 |
|---|---|---|
| Arranca en | segundo 23 | segundo 4 |
| s (p5–p95) | 0.54–1.79 | 0.46–1.02 |
| NIS medio posición / velocidad / curso (esperado 2 / 1 / 1) | 0.50 / 1.79 / 0.72 | 0.17 / 0.86 / 0.41 |
| Velocidades rechazadas | 8 de 226 | 3 de 216 |
| Predicción contra cada fix nuevo (1 s de cámara), p50 / p95 | 1.8 / 7.2 m | 1.4 / 3.6 m |
| Cámara sola desde el mismo inicio (modelo plano, s = 1) | 45 % de lo recorrido | 45 % |

Barrido de cortes simulados (uno cada 10 s; solo ventanas a más de 3 m/s de media,
19–22 por caso; error al volver el GPS, p50 / p90). Script temporal; el barrido
formal va con la evaluación (parte 4):

| Corte | 14:05 fusión | 14:05 GPS a velocidad constante | 14:18 fusión | 14:18 GPS a velocidad constante |
|---|---|---|---|---|
| 10 s | 26 / 52 m (24 % de lo recorrido) | 35 / 55 m | 12 / 20 m (19 %) | 15 / 37 m |
| 20 s | 51 / 117 m (26 %) | 73 / 187 m | 18 / 44 m (17 %) | 39 / 90 m |
| 30 s | 79 / 152 m (30 %) | 65 / 310 m | 30 / 75 m (17 %) | 72 / 159 m |

Con el GPS congelado el error es casi lo recorrido. La fusión le gana a la velocidad
constante en 50–74 % de los cortes y sobre todo en el p90 (las curvas); en las
rectas la velocidad constante es casi perfecta. Una primera versión del barrido
(cada 20 s, n = 7–10) había dado 9/29/66 m en 14:05: con pocos cortes cambia mucho.

**De dónde sale el error en los cortes** (10 s): a lo largo del camino 18 m en
14:05 y 6 m en 14:18; de costado 5 y 4 m; rumbo ~7.5° al final, lo que predice el
error contra el giroscopio. **Domina la escala:** la s al empezar el corte difiere
16–19 % (mediana) de la que hace falta durante el corte. La escala del modelo de
profundidad cambia con la velocidad y la escena: en 14:05 va de 0.5 a 2.4, y a más
de 14 m/s la cámara mide la mitad de la velocidad (la U de §3.3). Con GPS, s la
sigue; en un corte, el filtro no puede anticiparla.

**La cámara sola da 45 %, no el 27–41 % de §4.13:** mismos datos, otra integración.
El modelo plano usa el paso de profundidad tal cual (en 14:18, 2083 m contra 1323 m
del GPS: +57 %) con el rumbo bien. En la cadena 3D de `evaluate.py`, la deriva de
cabeceo deforma el trazo, y por casualidad terminaba más cerca. **Para el OE4, el
modo solo visual es este** (el EKF sin GPS).

**Costo:** ~20 µs por frame (7800 frames en 0.15 s), despreciable.

**Falta del EKF:**
- ~~La compuerta de detenido~~ Hecha (§4.15).
- La integración al pipeline: `GpsBuffer` tiene que llevar también la velocidad
  Doppler.
- El modo solo visual sin GPS, para el OE4.
- La evaluación formal (RPE y barrido de cortes en `evaluate.py`).

**Condiciones del estudiante al aprobar las cuatro decisiones (4/10/2026):** reportar
siempre (a) cuánto cambia el rumbo con y sin la medida directa del curso y (d)
cuántas correcciones se rechazan y cuántas se fuerzan. `evaluate.py --fusion` las
imprime en cada corrida. En los fragmentos: (a) el rumbo difiere 1.6–1.7° (p50),
6° (p95), 11–12° como máximo; incertidumbre ~3.5° con el curso y ~5° sin él. (d)
posición 0 rechazos; velocidad 8 rechazos y 2 forzadas (14:05), 3 y 0 (14:18);
curso 1 rechazo (14:05).

### 4.15 EKF, segunda parte: compuerta de detenido (4/10/2026)

**Datos:** las dos grabaciones completas (24 min, ~20 altos), porque los fragmentos
casi no tienen altos (19 s y 9 s). Referencia: el Doppler, con su atraso; en los
altos largos el teléfono queda quieto (residuo ~1 px).

**Señales probadas** (la sonda reproduce el front-end al 100 %):

| Señal | Detenido | En marcha | Veredicto |
|---|---|---|---|
| Razón inliers/matches, frame a frame | p50 0.003–0.005 | 1–3 m/s: 0.012–0.05 | No separa por debajo de 3–5 m/s: con el capó tapado no hay puntos a menos de 50 desplazamientos |
| Flujo frame a frame, crudo y sin el giro | 1.0 y 0.8–1.0 px | 2–3 m/s: 2.0 y 1.5 px | ORB cuantiza a píxeles enteros y a 30 fps el avance por frame es subpíxel |
| Flujo crudo entre frames a 0.33 s | p50 5–9 px, p95 50–60 | 1.5–3 m/s: 12–22 px | El teléfono gira en los altos (cabeceo al frenar y arrancar, soporte): 10–20°/s en el giroscopio |
| **Residuo tras el mejor giro puro, a 0.33 s** | **p50 1.1–1.3 px, p95 2.1–2.5** | 1–1.5 m/s: p50 3.0; >5 m/s: 7.6–8 | **Separa** |

El residuo: se ajusta la rotación que mejor explica el movimiento de los puntos
(Kabsch sobre los rayos, reajustada con la mitad mejor) y se mide lo que sobra. Un
giro sin avance no deja paralaje. Verificado con datos sintéticos: quieto 1.2 px,
giro puro de 5° 1.2 px, avance a 1 m/s 2.3 px, a 10 m/s 19 px.

**Regla:** detenido si el residuo baja de 2 px dos veces seguidas y la velocidad
estimada es menor que 3 m/s; sale cuando pasa de 3 px. El límite de velocidad hace
falta: en autopista con escena lejana y camino mojado (14:18, 264–268 s a 17 m/s) el
residuo baja a 1.5–2 px durante 4 s, y un carro no se detiene tan rápido.

**Dónde va:** en `DepthScaleEstimator` (`rotation_residual`, `_update_stopped`),
que ya compara frames a 0.33 s. Detenido, reporta 0 m/s y vacía la mediana móvil
(la historia previa no dice nada del arranque). El EKF no necesita código propio:
paso 0 y sin corrección de escala. De paso se ahorra la inferencia del modelo con
el carro quieto. **El rumbo no se congela en los altos:** la VO coincide ahí con el
giroscopio (14:18, con Doppler < 0.3: −16.2° contra −16.4°).

**Resultados** (replays deterministas de las grabaciones completas, en
`resultados/realtime/full_14_05_{sin_compuerta,compuerta}/` y `full_14_18_...`,
con `evaluate.log` y `fusion.png`):

| | 14:05 | 14:18 |
|---|---|---|
| Velocidad de la cámara con el GPS detenido (p50) | 0.64 → 0.00 m/s | 0.44 → 0.00 m/s |
| Altos detectados (frames con el GPS detenido) | 85 % | 97 % |
| Frames a más de 1 m/s marcados detenidos | 1.1 % (9 m perdidos en 12 min) | 0.6 % (16 m) |
| Avance falso eliminado | 92 m | 38 m |

Barrido de cortes (uno cada 10 s, ~70 por caso), mediana del error al volver el GPS:

| Corte | 14:05 con un alto (sin → con compuerta) | 14:18 con un alto | 14:05 todos | 14:18 todos |
|---|---|---|---|---|
| 10 s | 3.4 → 2.5 m | 4.8 → 3.6 m | 8.3 → 7.7 m | 21.2 → 19.3 m |
| 20 s | 6.8 → 5.3 m | 9.9 → 9.4 m | 16.9 → 17.6 m | 41.4 → 39.5 m |
| 30 s | 8.4 → 8.1 m | 21.7 → 17.6 m | 33.5 → 34.5 m | 57.4 → 55.0 m |

En los cortes con un alto la fusión le gana a la velocidad constante del GPS en
71–94 %. Cortes de ejemplo (elegidos de antemano sobre altos): 14:05 en 350 s (40 s)
14.7 → 11.4 m y en 575 s (30 s) 7.9 → 0.7 m; 14:18 en 685 s 9.1 → 6.5 m y en 615 s
6.6 → 4.9 m. La deriva final de la cámara sola en 12 min no sirve para medir la
compuerta: 56 → 58 % y 85 → 85 %, la domina el rumbo.

(a) y (d) en las grabaciones completas, con compuerta: el rumbo con y sin curso
difiere 0.7–1.6° (p50), 7° (p95), 16–22° como máximo; incertidumbre 3.5–5.2° con el
curso y 5.3–6.4° sin él. Rechazadas / forzadas: 14:05 posición 0/0, velocidad 11/2,
curso 1/0; 14:18 posición 6/2, velocidad 25/5 (más 19 fuera de rango), curso 2/1.

**Hallazgo: en la autopista de 14:18 el modelo de profundidad falla como sensor de
velocidad.** Entre 290 y 345 s, con el carro parejo a 13–20 m/s, la cámara dice de
0.2 a 32 m/s (razón Doppler/cámara hasta 57; p95 de 18 a 10–14 m/s). A los 341 s
dijo 0.2 m/s a 13 m/s: la aceptación forzada llevó s a 51 y la predicción a 128 m
del fix siguiente. **Red de seguridad agregada:** s queda entre 0.2 y 5 (medido
0.3–3.3) y una velocidad de la cámara que implica una s fuera de ese rango no se
usa ni cuenta para forzar. El peor error bajó a 42 m; los rechazos de posición que
quedan son todos de ese tramo. En la grabación completa de 14:18 la fusión le gana
a la velocidad constante solo en 44–63 % de los cortes (a 10 s, 19 m contra 17 m):
mientras la velocidad de la cámara sea una entrada de la predicción, si falla, la
predicción falla. **Decisión pendiente del estudiante:** pasar la velocidad a
estado (`[x, y, θ, v, s]`), con la cámara como medida de `v/s` que se puede
rechazar; con la cámara rechazada, el filtro sigue con la velocidad que traía (como
el GPS a velocidad constante) y el rumbo de la cámara. Los fragmentos de §4.14 no
cambian con la red de seguridad (nunca se activa ahí).

### 4.16 El lazo de la cámara sola en autopista: diagnóstico (4/10/2026)

Pregunta del estudiante: qué tan viable es arreglar el trazo solo visual a alta
velocidad (el lazo de 14:18). Todo medido con scripts temporales, sin tocar código.

**Causa: el rumbo, no la escala.** Integrando en el plano, desde el inicio de la
fusión, con cada combinación de giro y velocidad (separación final, % de lo
recorrido):

| | 14:05 | 14:18 |
|---|---|---|
| Cámara sola (giro y velocidad de la cámara) | 58 % | 85 %, con el lazo |
| Giro de la cámara + velocidad del GPS | 59 % | 72 %, con el lazo |
| Giroscopio + velocidad de la cámara | 11 % | 10 %, la forma correcta |
| Giroscopio + velocidad del GPS | 11 % | 3 % |

El error de rumbo de 14:18 es una rampa continua en la autopista: +230° entre 250 y
380 s, +76°/min a 14–18 m/s. Contra el giroscopio, la VO mide los giros ~15 % más
grandes (ganancia 1.14–1.16 en las dos grabaciones) y suma un sesgo de +0.5°/s aun
en recta.

**Hipótesis descartadas** (tramo de autopista de 14:18, 200–380 s, y urbano de 14:05,
40–200 s; deriva de rumbo contra el giroscopio):

| Variante | 14:05 | 14:18 |
|---|---|---|
| Como está | +19°/min | +55°/min |
| Distorsión corregida (coeficientes de §5.4 de `ESTADO_APP_ANDROID.md`) | +20 | +53 |
| Centro óptico en el centro de la imagen | — | +51 |
| Focal de 950 / 1000 / 1050 px (la ganancia baja a 1.07–1.10) | +15 a +17 | +49 a +55 |
| Máscara del 40 % (sin el asfalto mojado) | — | +61 |

La app no activa la estabilización de video (revisado en su código, solo lectura).

**Lo que sí funciona: el giro entre frames separados.** Mismo front-end, estimando
la pose entre un frame y el de 3 o 10 frames antes: cada 10 frames (0.33 s), 14:05
pasa a +4.5°/min (máximo 17° contra 55°) y 14:18 a +30°/min (máximo 98° contra 175°).
Cada 3 frames empeora la autopista (+73°/min): no es monótono, el tramo tiene algo
propio (candidatas sin medir: tránsito que se mueve con el carro, el camino mojado).
Sobre las grabaciones completas, con la velocidad de la cámara con compuerta:

| Cámara sola | Giro frame a frame (hoy) | Giro cada 10 frames |
|---|---|---|
| 14:05, error máximo de rumbo | 103° | 48° |
| 14:05, separación a 120 s / final | 423 m / 58 % | 85 m / 14 % |
| 14:18, error máximo de rumbo | 227° | 116° |
| 14:18, separación a 120 s / final | 475 m / 85 % | 277 m / 34 %, sin lazo |

Figuras en la carpeta temporal (se pierden); los números quedan aquí.

**Opciones evaluadas** (costo, alcance, RL):
- **Giro entre frames separados (recomendada entonces; descartada el 7/10 como fuente
  principal):** 1–2 días (front-end, barrido de la
  separación con las dos grabaciones, replays, volver a medir el ruido de rumbo del
  EKF y evaluar). No cambia el alcance. No interfiere con el RL si se hace antes de
  rehacer la puerta del OE3: cambia el ruido que ve el filtro, así que tiene que
  quedar fijo antes de entrenar.
- Calibración (distorsión, centro, focal): medida, no arregla el lazo.
- Giroscopio: el mejor rumbo (10–11 %), pero cambia el alcance (el anteproyecto dice
  sin IMU; pregunta abierta con el asesor, §11) y la definición del OE3 y del OE4.
  **Elegida el 7/10/2026** (abajo).
- Optimización por ventanas (ajuste de haces) o seguimiento KLT: semanas, riesgo
  alto; el asesor dijo que no se exige.
- La velocidad como estado (decisión de §4.15) no toca el lazo: ataca la velocidad
  errática, no el rumbo. Es complementaria.

**Decisión (7/10/2026): el rumbo sale del giroscopio.** El estudiante lo consultó con
el asesor, que lo aprobó. **El giro cada 10 frames queda descartado como fuente
principal**: se haría al final, solo si sobra tiempo, como respaldo y para la línea
base solo visual. Medido para decidir, sobre las grabaciones completas (giro alrededor
de la vertical que da el acelerómetro; el teléfono queda a 2.1° y 4.6° de su eje x):

| | 14:05 | 14:18 |
|---|---|---|
| Rumbo del giroscopio solo contra el curso del GPS (>5 m/s), tendencia | −1.7°/min | −1.2°/min |
| Error p50 / p95 / al final de los 12 min | 3.7° / 18.7° / −12° | 3.3° / 10.2° / −6° |
| Giro medido en cada alto (Doppler < 0.15 m/s durante 6 s o más) | −0.26 a +0.27°/s | −1.35 a +0.44°/s |

- En un corte de 30 s el giroscopio solo pierde menos de 1°. La cámara deriva de 3 a 25
  veces más aun cada 10 frames, así que no puede corregirlo; lo corrige el curso del GPS,
  que el EKF ya usa. **El sesgo del giroscopio va como estado**, estimado con el curso.
- Los altos no sirven para medir el sesgo: el soporte se mueve y algunos "altos" según
  el Doppler son maniobras lentas.
- Sin muestras del giroscopio (app sin IMU, paquetes perdidos), el EKF usa en ese frame
  el giro de la cámara, con la red de 10°.
- El alcance declarado ("sin IMU") deja de regir; los objetivos específicos no cambian.
  El margen del RL se achica, y el estudiante lo acepta: su meta es que el RL cumpla lo
  mínimo declarado (darle más peso a un sensor u otro según la situación).
- **La velocidad pasa a estado** (`[x, y, θ, v, s]` más el sesgo), decidido el mismo
  día: con el rumbo resuelto, el error que queda es la velocidad (§4.15). Se mide con y
  sin, igual que el sesgo.
- Para el OE4 se reportan cuatro modos: cámara sola, cámara + giroscopio, cámara + GPS
  y los tres juntos, para que se vea qué aporta cada sensor.

### 4.17 EKF con el giroscopio y la velocidad como estado (7/10/2026)

**Dónde está:** `realtime/fusion.py` y `evaluate.py --fusion --sweep`. Todavía sin
conexión (sobre los `frames.csv` guardados); el pipeline es el paso siguiente.
Resultados en `resultados/realtime/giro_14_05/` y `giro_14_18/` (`evaluate.log`,
`evaluation.json`, `fusion.png`), sobre las grabaciones completas con la compuerta.

**Diseño:**
- **Estado** `[x, y, θ, s, v]`, más el sesgo `b` del giroscopio como opción. Predicción
  en cada frame: rumbo += giro del giroscopio alrededor de la vertical (la vertical es
  un pasabajos de 20 s del acelerómetro; `GyroYaw`, que funciona muestra a muestra como
  en vivo); posición += v·dt. Si un frame no tiene muestras del giroscopio, usa el giro
  de la cámara (el giroscopio cubrió el 100 % de los frames en las dos grabaciones).
- **Correcciones:** con cada fix, posición, Doppler (= v, a cualquier velocidad) y curso;
  cada segundo, la velocidad de la cámara como medida de `v/s` (nunca forzada: si no
  coincide, el filtro sigue con la velocidad que traía), o v = 0 cuando la compuerta de
  detenido ve el carro quieto. Una velocidad de la cámara que implica s fuera de 0.2–5
  no se usa.
- **Sin GPS, s queda congelada** (ni deriva ni se corrige) a los 2 s del último fix. Sin
  esa regla la velocidad como estado fallaba: sin una velocidad de referencia, cada
  cambio de velocidad que veía la cámara se repartía entre v y s, y s se iba a 0.26
  (14:05); con cámara y giroscopio sin GPS daba 53 % de lo recorrido. Medido: entre la
  cámara y el Doppler la escala cambia ~45–56 % en 5 s y después casi nada en 14:05
  (52 % a 30 s, 50 % a 120 s); no es un paseo aleatorio a plazos largos.
- **Modo solo visual** (`visual_only`): el GPS solo ubica el arranque (dónde y hacia
  dónde); después la cámara (y el giroscopio, si está) con la escala propia del modelo
  (s = 1). Es la línea base del OE4 y reemplaza el truco de cortar el GPS después de
  arrancar, que en el filtro nuevo dejaba aprender s de la primera medida.
- Con giroscopio, velocidad y sesgo apagados, el código nuevo reproduce el filtro
  anterior hasta el redondeo (1e-13 m).

**Ruidos nuevos, medidos antes de correr el filtro:**

| Parámetro | Valor | Medición |
|---|---|---|
| Velocidad como paseo aleatorio | 1 m/s por √s | función de estructura del Doppler: 0.3, 0.9, 1.7, 3.7 y 9.5–10.6 m²/s² a 1, 2, 3, 5 y 10 s |
| Velocidad con la compuerta en detenido | σ 0.5 m/s | Doppler p50 0, p95 0.94–0.99 m/s; >3 m/s en 0.3 % (14:05) y 1.2 % (14:18, la autopista) |
| Rumbo del giroscopio | 0.5°/√s (tope) | residuo contra el curso del GPS (>8 m/s) tras quitar la tendencia: crece ~0.3 grados²/s; incluye los errores lentos del propio GPS |
| Sesgo del giroscopio | inicial 0.07°/s; 0.003°/s por √s | tendencia −0.035 y −0.009°/s; por ventanas de 2 min, de −0.03 a +0.07°/s |
| Velocidad de la cámara | 20 % + 0.5 m/s, cada 1 s | los de §4.14, más el piso de §3.3 (error ~57 % a 0–2 m/s) |

**Barrido de cortes** (uno cada 10 s en toda la grabación, 57–75 por duración; error al
volver el GPS, p50 / p90 en metros):

| 14:05 | 10 s | 20 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|
| Filtro anterior (cámara) | 7.5 / 45 | 21.0 / 102 | 31.6 / 143 | 73 / 250 | 148 / 426 |
| Cámara, con velocidad como estado | 7.4 / 31 | 21.9 / 69 | 33.9 / 129 | 67 / 238 | 155 / 430 |
| Giroscopio, sin velocidad como estado | 6.7 / 34 | 14.0 / 63 | 19.0 / 95 | 24 / 170 | 69 / 373 |
| **Giroscopio + velocidad como estado** | **4.9 / 23** | **9.8 / 49** | **15.7 / 79** | **25 / 186** | **61 / 318** |
| ... + sesgo | 4.9 / 23 | 10.1 / 49 | 16.4 / 79 | 31 / 187 | 60 / 318 |
| GPS a velocidad constante | 13.2 / 40 | 30.3 / 113 | 53.8 / 192 | 110 / 462 | 357 / 1012 |

| 14:18 | 10 s | 20 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|
| Filtro anterior (cámara) | 18.3 / 89 | 32.9 / 185 | 44.4 / 336 | 127 / 919 | 395 / 2194 |
| Cámara, con velocidad como estado | 14.5 / 60 | 26.3 / 155 | 52.2 / 302 | 118 / 843 | 421 / 2121 |
| Giroscopio, sin velocidad como estado | 16.0 / 89 | 29.2 / 149 | 41.6 / 250 | 92 / 506 | 171 / 1526 |
| **Giroscopio + velocidad como estado** | **10.6 / 43** | **16.9 / 107** | **27.3 / 178** | **83 / 447** | **178 / 1348** |
| ... + sesgo | 11.3 / 43 | 16.6 / 106 | 27.5 / 178 | 85 / 447 | 198 / 1348 |
| GPS a velocidad constante | 21.1 / 45 | 46.1 / 123 | 79.1 / 254 | 224 / 651 | 693 / 1268 |

- **Le gana al filtro anterior en todas las duraciones de las dos grabaciones** (la
  condición para seguir). Gana a la velocidad constante en 72–96 % de los cortes en
  14:05 y 65–80 % en 14:18.
- **El giroscopio** quita el error de rumbo: es lo que más baja los cortes largos.
- **La velocidad como estado** corta los peores casos (p90 a la mitad en 10 s): la
  velocidad de la cámara que falla en la autopista se rechaza en vez de entrar a la
  predicción.
- **El sesgo no ayudó** en ninguna duración (igual o peor a 60–120 s), aunque su
  estimación coincide con la tendencia medida (+0.044 y +0.035°/s). Queda apagado por
  defecto y disponible para la comparación.
- En 14:18 a 120 s, el p90 sigue por encima de la velocidad constante (1348 contra
  1268 m): los cortes largos en el tramo de autopista, donde la cámara falla.

**Lo demás, con la configuración elegida (giroscopio + velocidad, sin sesgo):**

| | 14:05 | 14:18 |
|---|---|---|
| Predicción contra cada fix nuevo, p50 / p95 | 1.3 / 3.9 m | 1.3 / 3.3 m |
| (a) Rumbo con y sin curso: p50 / p95 / máx | 0.1 / 1.1 / 3.3° | 0.3 / 1.3 / 3.0° |
| (a) Incertidumbre del rumbo con / sin curso | 1.9 / 2.1° | 1.5 / 1.7° |
| (d) Posición, rechazadas / forzadas | 0 / 0 de 695 | 0 / 0 de 767 |
| (d) Doppler | 0 / 0 de 695 | 1 / 0 de 767 |
| (d) Curso | 3 / 0 de 351 | 0 / 0 de 588 |
| (d) Velocidad de la cámara (nunca se fuerza) | 0 de 553 | 9 de 660, más 18 fuera de rango |
| (d) Detenido (v = 0) | 0 de 139 | 1 de 82 |
| Cortes de ejemplo de §4.15 | 350 s: 11.4 → 3.0 m; 575 s: 0.7 → 1.5 m | 685 s: 6.5 → 0.2 m; 615 s: 4.9 → 3.9 m |
| Solo visual, cámara sola (al final) | 55 % de lo recorrido | 86 % |
| Solo visual, cámara + giroscopio | 5 % | 18 %, sin el lazo |

Con el giroscopio el curso del GPS casi no cambia el rumbo (0.1–0.3° de mediana): el
giroscopio ya lo sostiene. Antes difería 0.7–1.6°. Los rechazos de posición del filtro
anterior en 14:18 (6, más 2 forzados) desaparecen. Los NIS medios quedan por debajo de
lo esperado (posición 0.17–0.19, Doppler 0.36–0.50): el filtro es conservador, como el
anterior. Sin GPS en toda la grabación, el filtro sin velocidad como estado da 11 % y
11 % con cámara y giroscopio: mismo orden que con ella.

### 4.18 Verificación del EKF: NIS y fixes absurdos (8/10/2026)

Pedida por el estudiante antes de integrar. Scripts temporales (no quedan en el repo);
figuras y números en `resultados/realtime/giro_verificacion/` (`nis.png`,
`rechazo.png`, `verificacion.json`, `arreglos_candidatos.json`, `arreglo_c.json`).

**NIS en marcha normal** (media; entre paréntesis, % sobre el percentil 95, que debería
ser 5 %):

| | 14:05 anterior | 14:05 nuevo | 14:18 anterior | 14:18 nuevo |
|---|---|---|---|---|
| Posición (esperado 2) | 0.36 (0.4 %) | 0.19 (0 %) | 0.97 (3.0 %) | 0.17 (0 %) |
| Doppler (esperado 1) | 1.53 (7.1 %) | 0.36 (0.4 %) | 2.33 (10.1 %) | 0.50 (0.5 %) |
| Curso (esperado 1) | 0.83 (5.1 %) | 0.60 (3.1 %) | 0.72 (2.4 %) | 0.31 (1.0 %) |
| Velocidad de la cámara | — | 0.56 (2.2 %) | — | 0.95 (6.4 %) |
| Detenido | — | 0.36 (1.4 %) | — | 1.47 (1.2 %) |

**El giroscopio y la velocidad no lo acercan: lo bajan más**, porque la predicción mejoró
(1.3 m a 1 s) y la σ del GPS sigue en 4 m. Es baja por construcción: el error del GPS
del teléfono es un sesgo que deriva despacio (de fix a fix es suave), y 4 m describe su
exactitud absoluta, no su ruido de fix a fix. Con σ de 1, 1.5, 2 y 4 m el NIS de
posición pasa de 0.9 a 0.2, la predicción a 1 s de 0.8 a 1.3 m (el filtro pega más o
menos al GPS) y **los cortes casi no cambian** (14:05 a 10/30/120 s: 4.6/15.5/64 contra
4.9/15.7/61 m; 14:18: 11.5/26.5/190 contra 10.6/27.3/178 m), sin rechazos ni forzados
con ninguna. Se queda en 4 m: una σ más chica rechazaría los saltos normales del GPS.

**NIS al volver el GPS después de un corte: ahí el filtro es demasiado confiado.**

| Corte | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| Error real / σ del filtro (p50), 14:05 y 14:18 | 2.1 y 2.5 | 3.0 y 3.3 | 3.3 y 5.2 | 3.9 y 6.1 |
| Primer fix bueno rechazado | 10 y 16 % | 30 y 42 % | 40 y 61 % | 58 y 72 % |

Al volver de un corte largo, el primer fix bueno se rechaza y entra forzado dos fixes
después. Causa: con la escala congelada, la velocidad sigue a la cámara como si su
escala fuera exacta, pero la escala congelada difiere ~18 % de la que hace falta durante
el corte (16–19 %, §4.14), y ese error crece con la distancia recorrida. Dos arreglos
probados:
- **A, la escala congelada como estado "considerado"** (su incertidumbre entra en la
  corrección, su valor no cambia): consistente (error/σ 1.2 a 10 s), **pero empeora el
  trazo** (14:05 a 120 s: 61 → 117 m; solo visual con giroscopio: 5 → 46 % y 18 → 67 %),
  porque la cámara pierde peso sobre la velocidad. Descartado.
- **C, el desajuste de escala (18 %) sumado solo a la incertidumbre de la posición a lo
  largo del camino**, creciendo con los metros recorridos desde que se congeló: **el
  trazo queda idéntico** (mismos cortes y mismo solo visual) y la incertidumbre se vuelve
  consistente: error/σ 1.2–1.3 (10 s), 1.3–1.4 (30 s), 1.5–1.7 (60 s), 1.3–2.0 (120 s);
  primer fix rechazado 0–10 % en todas las duraciones. **Propuesto.**

**Fixes absurdos** (20 puntos en marcha por grabación, dirección al azar):

| Prueba | Resultado |
|---|---|
| 1 fix a 1 km | 40 de 40 rechazados; el trazo se mueve 0.3–0.5 m (p50), 3 m como máximo |
| 2 seguidos a 1 km | 80 de 80 rechazados |
| 3 seguidos a 1 km | el tercero entra forzado (40 de 40): el trazo salta 1 km ~3 s y vuelve en menos de 30 s |
| 5 seguidos a 1 km | el tercero forzado y los siguientes aceptados: el trazo sigue al GPS absurdo |
| 1 fix a 20 / 50 / 100 m | 39 de 40 / todos / todos rechazados (el umbral es ~17 m) |
| 1 km como primer fix después de 30 o 120 s sin GPS | 80 de 80 rechazados |

**La compuerta rechaza los saltos sueltos, pero la aceptación forzada deja pasar un
salto que dura 3 fixes o más.** Arreglo probado, **B**: un fix de posición solo se fuerza
si el carro pudo llegar ahí desde el último fix aceptado (40 m/s, ~144 km/h, más 3σ del
GPS). Con B, 3, 5 y 10 fixes seguidos a 1 km se rechazan todos (0 forzados; el trazo se
mueve menos de 5 m), y nada más cambia (cortes, NIS, solo visual iguales). Sigue
dejando forzar al volver de un corte largo, donde el error real cabe en el alcance
(120 s × 40 m/s). **Propuesto.**

**B y C implementados el 8/10/2026** (aprobados por el estudiante), en `fusion.py`:
`PlanarEKF.HELD_SCALE_ERR` (C, en `predict`) y `GpsCameraFusion.V_MAX` (B, en
`on_fix`, con `position_innovation` separado de `update_position`). Al verificarlos
apareció un hueco en B tal como se había probado: si el primer fix después de un corte
largo era absurdo y entraba por la compuerta (la incertidumbre es grande ahí; 1 de 20
casos en cada grabación), el alcance se medía desde ese fix, los fixes buenos quedaban
"inalcanzables" y el trazo seguía a 1 km a los 10 s (se habría recuperado a los ~25 s).
**Ajuste:** el alcance se cuenta desde el último fix que coincidió con la predicción
(innovación menor que 3σ del GPS); un fix aceptado con una innovación grande no lo
mueve. Resultados con el código final (scripts temporales; `con_arreglos.json` y
`con_arreglos.png` en `resultados/realtime/giro_verificacion/`):
- **La evaluación completa es idéntica a la del paso 1**, byte a byte (correcciones,
  NIS, predicción, solo visual, cortes de ejemplo y el barrido entero): ni B ni C
  cambian la estimación.
- **Al volver de un corte:** error/σ 1.2–1.3 (10 s), 1.3–1.4 (30 s), 1.5–1.7 (60 s),
  1.3–2.0 (120 s), contra 2.1–6.1 antes; primer fix bueno rechazado 0–10 %, contra
  10–72 %.
- **Fixes absurdos:** 1, 2, 3, 5 y 10 seguidos a 1 km, todos rechazados y ninguno
  forzado; el trazo se mueve como mucho 8 m y a los 3 s del último está a menos de 7 m
  de la corrida limpia. Saltos sueltos de 20, 50 y 100 m, todos rechazados.
- **1 km justo al volver de un corte:** rechazado en 79 de 80 casos. Retrasa un fix la
  vuelta al GPS (durante ese segundo el trazo sigue con el error acumulado del corte:
  ~100 m p50 a 120 s) y a los 3 s está a menos de 4 m. El caso que entra se corrige en
  ~3 s (a 1–2 m de la corrida limpia), donde antes del ajuste quedaba a 1 km.

### 4.19 El EKF dentro del pipeline (8/10/2026)

**Estado: terminado y verificado** (replay determinista y estricto, en vivo con el
simulador con y sin IMU, y la ventana). **Desde el 9/10/2026 corre en vivo con la IMU
del teléfono real** (§4.20); sin ella el filtro corre igual, con el giro de la cámara.

**Qué cambió** (paso 2 del plan):
- `fusion.OnlineFusion`: la fusión dentro del pipeline, un frame procesado a la vez.
  Toma las muestras de la IMU y los fixes que llegaron hasta ese frame y avanza el
  filtro en el mismo orden que `run_fusion` (el frame predice, los fixes hasta su
  marca de tiempo corrigen). Corre en el hilo de procesamiento, después del front-end
  visual. `correction_counts` es el conteo de correcciones compartido con `evaluate.py`.
- `sources.py`: `GpsBuffer` guarda la velocidad Doppler de cada fix y entrega los
  nuevos en orden (`between`); `ImuBuffer` nuevo (cada muestra se toma una vez);
  `ReplayFeed` reemplaza a `ReplayGpsSource` y reproduce fixes o muestras de la IMU en
  tiempo real; `LiveGpsSource` lee las líneas `IMU` del mismo socket UDP (formato del
  pedido a la app, §10); `load_mobile_imu` lee `gyro_accel.csv`.
- `pipeline.py`: `RealtimePipeline(fusion=...)`; su tiempo por frame va a las métricas
  (`fusion_ms_p50/p95`).
- `frames.csv` agrega `gyro_yaw` (radianes; vacío si giró la cámara) y el estado
  después de cada frame: `fus_x`, `fus_y`, `fus_heading`, `fus_s`, `fus_v`,
  `fus_sigma_m` (vacíos antes del arranque). El gráfico de cada sesión muestra el trazo
  fusionado. `metrics.json` trae el bloque `fusion`.
- `run_replay.py` y `run_live.py`: la fusión corre siempre; `--visual-only` (el GPS solo
  para arrancar, la línea base del OE4) y `--no-gyro` (girar con la cámara).
  `session.py`: las mismas dos opciones (`visual_only`, `gyro`); en vivo guarda la IMU
  recibida como `gyro_accel.csv`, el mismo archivo de las grabaciones, así que
  `evaluate.py` analiza la sesión con el giroscopio; `status()` da `fusion_sigma_m`.
- `phone_simulator.py` manda las líneas `IMU` como las mandará la app; `--no-imu` la
  deja afuera, como la app actual.

**Verificación** (scripts temporales; sesiones en `resultados/realtime/integracion_14_05/`
e `integracion_vivo/`):
- **Replay determinista** (14:05, segundos 100 a 200, con profundidad): la fusión dentro
  del pipeline da lo mismo que `run_fusion` sobre el `frames.csv` guardado: posición a
  0.5 mm como máximo (0.04 mm p50), rumbo a 3·10⁻⁵°, el giro del giroscopio por frame a
  5·10⁻¹¹ rad; las mismas correcciones. Las diferencias son el redondeo del CSV.
- **Costo:** 0.13–0.14 ms por frame (p50), 0.19–0.25 ms (p95).
- **En vivo con el simulador** (14:05 desde el segundo 100, 75 s, profundidad): 4177
  muestras de IMU (53 Hz), 100 % de los frames girados con el giroscopio, posición
  73/0/0. `evaluate.py` lee la carpeta de la sesión con su `gyro_accel.csv`. Contra el
  cálculo aparte: 0.03 m p50, 0.7 m como máximo (en vivo arranca 3 s antes: los fixes
  que llegan mientras carga el modelo ya sirven); el giro del giroscopio, idéntico.
- **Sin IMU** (`--no-imu`, como la app actual): la fusión sigue con el giro de la cámara
  y el resumen lo dice. `--visual-only`: s queda en 1.00 y solo corrige la cámara.
  `--no-gyro`: gira la cámara y s se aprende del GPS.
- **La ventana en Xvfb**, tres sesiones seguidas (en vivo con profundidad e IMU, en vivo
  grabando, replay con el giroscopio de la grabación): la fusión en marcha en las tres
  (σ 1.4–1.9 m), 100 % con giroscopio, `frames.csv` con el estado, sin errores de Tk,
  una sola ventana.
- La evaluación completa de §4.17 sale idéntica; invariantes 6/6.

**Bug encontrado y corregido:** el replay determinista cargaba todos los fixes de
entrada en un `GpsBuffer` de 600 lugares; con 719–770 fixes se perdían los primeros.
Los `frames.csv` de `full_14_*_compuerta/` no tienen fix emparejado en los primeros
119 s (14:05) y 170 s (14:18) (`frames_sin_gps` 3570 y 5112). **No afecta §4.14–4.18**
(`evaluate.py` lee `location.csv`); sí las columnas de GPS de esos `frames.csv`, la
antigüedad del fix de esos replays y sus gráficos de trayectoria. Ahora el modo
determinista usa buffers sin límite.


### 4.20 La IMU del teléfono real, en vivo (9/10/2026)

La app manda la IMU desde el 9/10 (pedido 1 de §10, hecho). **Prueba 1: llegan los tres
sensores, en el mismo reloj, y el giroscopio da el giro de cada frame.** Teléfono en la
mano, en horizontal con la cámara a la izquierda, girado a los lados (de −149° a +229°
acumulados), bajo techo, por el WiFi de la casa (IP 192.168.100.113).

**Antes, la ventana:** dos líneas nuevas (y lo mismo al final de la línea de la consola):
- *IMU:* `53 muestras/s; el giro lo da el giroscopio`; `…, pero el giro lo da la cámara`
  si llegan muestras que no cubren el frame (un hueco, u otro reloj: el giroscopio solo
  gira un frame que sus muestras cubren, así que esta línea también delata un reloj
  distinto); `no llega; el giro lo da la cámara`.
- *Fusión:* `espera ver el carro avanzar 20 m a más de 3 m/s`, o `en marcha, posición ±
  X m`. Al terminar: el % de frames que giró el giroscopio y si arrancó.
- `OnlineFusion` cuenta los frames girados por el giroscopio **desde el primero**
  (`frames_con_imu_pct` en `metrics.json`): antes del arranque ya gira el rumbo con el
  que el filtro arranca. `frames_con_giroscopio_pct` sigue contando desde el arranque.
- Probado en Xvfb con el simulador (con IMU, con la IMU cortada, replay): todo bien.

**Resultados** (`run_live --record --scale depth --duration 60`, sesión
`resultados/realtime/live_20261009_190802/`; la grabación del teléfono de ese mismo
minuto, traída con `adb pull`, en `mobile_data/2026_10_09_19_08_06/`; scripts temporales):

| | Llegó en vivo | Grabó el teléfono en ese tramo |
|---|---|---|
| Frames | 1646 recibidos (8.5 % no enviados por el teléfono), 1596 procesados (3.0 % descartados) | 1798; los 1596 procesados están con la misma marca (±1 µs) |
| Fixes de GPS | 76 (1.19 Hz, bajo techo) | 71; los 71 llegaron (5 más antes de empezar a grabar) |
| Muestras de IMU | 3400 (53 Hz) | 3196; **las 3196 llegaron, ninguna perdida**, con los mismos valores; 201 más antes de empezar a grabar |

- IMU: `t_ns` siempre creciente, intervalo 18.9 ms (máximo 19.0 ms: ningún hueco).
- **El giroscopio giró el 99.9 % de los frames** (todos menos el primero, que no tiene
  uno anterior). La consola y la ventana lo dijeron durante toda la sesión.
- **El mismo reloj:** (1) con el desfase de la consulta de hora (prueba 2), los frames
  llegan 81 ms después de su captura y la IMU 41 ms después de su marca (mínimo 35):
  positivos y estables; (2) en el replay, el giro del giroscopio contra el de la cámara
  coincide mejor con un desfase de **−4 ms** (resolución 2 ms; con 0 ms la diferencia
  casi no cambia: 0.911 contra 0.908° RMS por frame). La cámara gira 1.07 veces el
  giroscopio; en los frames que giran (> 0.5°), la diferencia mediana es 0.16° sobre
  giros de 1.5°. Bajo techo solo 463 de 1800 frames tienen ≥ 30 inliers.
- **En vivo contra replay:** el giro acumulado del giroscopio difiere a lo sumo 1.27° en
  378° de recorrido. No es pérdida de datos: el mismo cálculo fuera de línea sobre las
  muestras recibidas da el giro en vivo a 3·10⁻⁸°, y arrancando los dos desde el inicio
  de la grabación coinciden a 5·10⁻⁵°. La diferencia sale de la vertical inicial (el
  filtro de gravedad de 20 s parte del primer acelerómetro: en vivo, 4 s antes de que
  empiece la grabación).
- **El filtro no arrancó**, como se esperaba en casa (necesita 20 m a más de 3 m/s): la
  comprobación del % girado desde el arranque y del trazo fusionado queda para la
  prueba de campo. Latencia en la PC 28/46 ms (p50/p95), 26.6 Hz efectivos.
- **La ventana en el escritorio**, con el teléfono, 20 s (antes probada igual en Xvfb):
  `IMU: 51–55 muestras/s; el giro lo da el giroscopio` en todo momento, `Fusión: espera
  ver el carro avanzar…`; al final `el giroscopio giró el 100 % de los frames` y `no
  arrancó`. Sin errores de Tk; ninguna ventana quedó abierta.

**Prueba 2: la consulta de hora en el pipeline** (pedido 2 de §10, hecho en la app el
9/10). La latencia desde la captura vuelve a ser confiable.
- `sources.PhoneClock`: `LiveGpsSource` manda `TIME,<hora Unix de la PC>` cada 0.5 s por
  el mismo socket UDP. Con cada respuesta, el reloj de arranque del teléfono se ubica en
  el punto medio de la ida y vuelta (error ≤ mitad de la ida y vuelta). **Se usa solo la
  consulta más rápida de los últimos 60 s**, no el promedio: una lenta esperó en algún
  lado (ahorro de energía del WiFi, una cola), no necesariamente igual en cada sentido.
  La ventana de 60 s sigue la deriva entre los dos relojes.
- `LiveFrameSource` pasa la captura de cada frame al reloj de la PC (`Frame.capture_ns`);
  el pipeline mide con eso la latencia de captura a resultado. Sin respuestas (una app
  vieja), vuelve a la hora Unix del teléfono y la consola lo avisa como antes.
- La consola y la ventana muestran la latencia desde la captura como número principal y
  la de la PC entre paréntesis; el veredicto juzga la de la captura cuando todos los
  frames tienen el reloj medido. `metrics.json` trae `en_vivo.reloj` (consultas, ida y
  vuelta, deriva, hora Unix del teléfono contra la PC) y `frames_con_reloj_medido`; la
  carpeta guarda cada consulta en `reloj.csv`.
- `phone_simulator.py` responde `TIME` como la app (su reloj de arranque es el de la
  grabación mientras reproduce).
- Probado con el simulador (consola, y la ventana en Xvfb): 98 consultas, ida y vuelta
  0.1 ms, captura a resultado 34/38 ms (el simulador "captura" al enviar). Invariantes 6/6.

**Con el teléfono real** (WiFi de la casa, 60 s, `resultados/realtime/live_20261009_191902/`):

| | p50 | p95 |
|---|---|---|
| Captura → llegada | 76 ms | 167 ms |
| Captura → resultado | 96 ms | 180 ms |
| Llegada → resultado (PC) | 20 ms | 26 ms |

- 125 consultas, todas respondidas; ida y vuelta mínima 2.2 ms (error ≤ 1.1 ms), p10
  2.9, mediana 5.4, p90 17, máximo 90 ms. Las 1706 frames con el reloj medido. La
  consulta más rápida de cada tramo de 10 s tuvo 2.2–3.0 ms.
- **Por WiFi el objetivo no se cumple** (p95 180 ms > 150): la mediana sí. El exceso
  es del enlace (captura a llegada p95 167 ms), no de la PC (26 ms). Confirma §4.9 ya
  sin el desfase. **Falta medirlo por cable** (anclaje USB), como irá en campo: el
  5/10 dio p95 113 ms, sin confirmar.
- En la ventana del escritorio con el teléfono (15 s): `latencia 98 ms (en la PC 19–25)`.
- **El desfase de relojes se mueve, y hoy fue la PC.** La hora Unix del teléfono menos
  la de la PC dio +132 ms a las 18:50 y +58 ms a las 19:19; dentro de la sesión subió
  2.4 ms en 60 s (~35 ppm). Pero la hora Unix del teléfono menos su reloj de arranque
  quedó igual a 0.01 ms entre las 18:50 y las 19:19: el teléfono no corrigió su hora.
  La PC sí: `systemd-timesyncd` contra `ntp.ubuntu.com` reporta un desfase de +68 ms,
  jitter de 117 ms y frecuencia corregida en +59.6 ppm (arrancó a las 18:18). Para la
  latencia da igual cuál se mueva: la consulta mide la relación entre los dos relojes
  en la misma sesión, y la ventana de 60 s sigue la deriva (≤ ~3.6 ms a 60 ppm).
- **11/10/2026: por cable, el objetivo de latencia se cumple** (anclaje USB, WiFi del
  teléfono apagado, IP 10.174.238.48, 60 s con escala por profundidad,
  `resultados/realtime/live_20261011_122734/`; la configuración de la prueba de campo):

  | | Cable USB (11/10) | WiFi de la casa (11/10, 30 s) |
  |---|---|---|
  | Frames | 1800 recibidos, 0 no enviados; 1799 procesados (0.1 % descartados), 30.0 Hz | 690, 22.8 % no enviados |
  | Captura → llegada p50/p95 | 70 / 76 ms | 81 / 186 ms |
  | **Captura → resultado p50/p95** | **97 / 107 ms: cumple (< 150)** | 112 / 224 ms: no cumple |
  | En la PC p50/p95 | 27 / 32 ms | 31 / 45 ms |
  | Consulta de hora | 126; ida y vuelta mínima 1.0 ms (error ≤ 0.5 ms), mediana 2 ms | 66; mínima 2.3 ms, mediana 7 ms |
  | GPS / IMU | 75 fixes (1.17 Hz) / 3398 muestras (53 Hz) | 13 (0.37 Hz) / 1801 (53 Hz) |

  Giroscopio en el 99.9 % de los frames en las dos. Hora Unix del teléfono −10 ms
  respecto de la PC (el 9/10 iba +58). `phone_check` por cable: 29.9 fps, IMU 52.9/s,
  frames 71 ms y IMU 35 ms después de su marca. **Confirma el OE1 por cable**: el
  p95 de 113 ms del 5/10, sin confirmar, queda en 107 ms medido. Por WiFi el exceso
  es del enlace (captura a llegada p95 186 ms), no de la PC.
- Usar la mediana de todas las consultas en vez de la más rápida habría dado +0.2 ms
  en esta sesión (consultas lentas simétricas); con la mediana de ida y vuelta de 53 ms
  que vio el task de la app, el error posible sube a ±27 ms.
- **11/10/2026: la laptop de campo, por cable, no cumple el p95.** Laptop: Intel
  i5-8300H (4 núcleos, 8 hilos, 2018), GTX 1050 Ti 4 GB (sm_61), driver 580, Linux
  Mint 22, `torch 2.8.0+cu126` (§15). Teléfono por anclaje USB en 10.174.238.48, bajo
  techo, en la mano, sin avanzar. `phone_check`: todo bien (30 fps, IMU 53/s con hueco
  máximo de 19 ms, ida y vuelta 2 ms; un primer intento sin la app en la pantalla de
  video dio "sin HELLO" e IMU a ráfagas). Dos sesiones desde la ventana, con escala:

  | | Sin grabar (`live_20261011_133826`, 54 s) | Grabando (`live_20261011_134001`, 52 s) | Escritorio, cable (`122734`) |
  |---|---|---|---|
  | Frames | 1606 recibidos, 0 no enviados; 1034 procesados (35.6 % descartados), 19.3 Hz | 1551; 978 (36.9 %), 18.9 Hz | 0.1 %, 30 Hz |
  | Captura → llegada p50/p95 | 67 / 73 ms | 66 / 73 ms | 70 / 76 ms |
  | **Captura → resultado p50/p95** | **138 / 167 ms: no cumple** | **132 / 189 ms: no cumple** | 97 / 107 ms |
  | En la PC p50/p95 | 70 / 99 ms | 66 / 122 ms | 27 / 32 ms |
  | Por frame p50 (ORB / emparejamiento / pose) | 52 ms (20 / 15 / 15) | 50 ms (19 / 14 / 15) | 24–27 ms |
  | Profundidad (GPU) p50/p95 | 291 / 401 ms, 154 estimaciones (~2.9 Hz) | 291 / 448 ms, 145 | 66–150 ms |

  Giroscopio en el 99.9 % de los frames; 73–74 fixes; 0 JPEG ni UDP inválidos; 113 y
  104 consultas de hora, todas respondidas. **El filtro no arrancó**, como se esperaba:
  el GPS se movió a lo sumo 1.0 y 1.9 m (velocidad máxima 0.26 y 0.49 m/s). Poses
  descartadas por giro > 10°: 15 y 22, e inliers p50 16–21: lo normal bajo techo sin
  avanzar (como el 6/10). La grabación del teléfono es `mobile_data/2026_10_11_13_40_02`
  (52 s, quieta; no sirve para medir el trazo).
  **El enlace es el mismo que en la de escritorio; el exceso es del cómputo.**
  Replay de esa grabación en la laptop (1561 frames; la latencia del replay equivale a
  la "en la PC" en vivo):

  | Replay | Por frame p50 | Latencia p50 / p95 | Descarte |
  |---|---|---|---|
  | Determinista, sin escala | 48.2 ms (p95 65) | — | (predice 26.9 %) |
  | Estricto, sin escala | 48.5 ms | 62 / 92 ms | 31.5 % |
  | Estricto, con escala | 51.8 ms | 66 / 100 ms | 37.6 % |
  | En vivo con la ventana (sesión 2) | 50 ms | 66 / 122 ms | 36.9 % |

  Por frame (~48 ms) es más que el intervalo entre frames (33 ms): siempre hay uno
  esperando. En vivo ≈ enlace (~70) + PC (~100) + ventana y decodificación (~20).
  Lo que **no** ayuda: hilos de OpenCV (8, 4, 2, 1: 48.5 / 49.4 / 54.9 / 80.3 ms por
  frame; ya usa 8); el CPU no se frena (~3.5 GHz medio, 73 °C, perfil `balanced`), así
  que "performance" ganaría poco; sin escala, el p95 en la PC solo baja de 100 a 92 ms.
  Lo único que lo mueve es el ORB (medido con un envoltorio, sin tocar el código;
  estricto con escala):

  | Puntos ORB | Por frame p50 (ORB / emparejamiento / pose) | En la PC p95 | En vivo p95 estimado |
  |---|---|---|---|
  | 2000 (actual) | 51.8 ms (19.4 / 16.0 / 14.8) | 100 ms | ~190 ms |
  | 1500 | 40.5 ms (18.0 / 9.2 / 12.1) | 88 ms | ~175 ms |
  | 1000 | 34.3 ms (18.2 / 4.7 / 10.5) | 76 ms | ~165 ms |

  La detección no baja con menos puntos (la pirámide y FAST dominan): para cumplir
  con holgura habría que bajar también la resolución. Las dos cosas separan el
  pipeline del de fuera de línea (invariante T6) y piden validarlas con grabaciones
  de manejo en la de escritorio.
  **Decisión del estudiante (11/10): la prueba de campo va con la configuración
  actual** (opción 1: frecuencia ~19 Hz, mediana ~135 ms, p95 ~170–190 ms en esta
  laptop; en la de escritorio cumple). **Sin medir:** el trazo y el EKF con ~37 % de
  frames descartados. Todo lo de §4 sobre datos de manejo se midió con ~1 %. El
  giroscopio cubre igual todos los frames (integra las muestras, no los frames) y el
  GPS corrige igual, pero el paso de la cámara entre frames procesados se duplica a
  veces. Se mide con `run_replay mobile_data/2025_03_11 --strict --scale depth
  --trajectory` en la laptop (contra 255 m y RMSE 17.45 m del determinista), o con la
  grabación de la prueba de campo.


### 4.21 Prueba de campo con la laptop (11/10/2026)

Primera prueba de campo: la laptop (§9) en el carro, el teléfono por cable (anclaje USB)
en el tablero, la ventana en vivo con escala y "Grabar también en el teléfono". Cuatro
sesiones en ciudad, de día, con lluvia intermitente. Todo medido con scripts temporales
(fuera del repo); los gráficos se pierden, los números quedan aquí.

| Sesión (`resultados/realtime/`) | Grabación (`mobile_data/`) | Duración | GPS recorrido | v máx | Procesados (descarte) | Hz | Captura → resultado p50/p95 |
|---|---|---|---|---|---|---|---|
| `live_20261011_145250` | `2026_10_11_14_52_51` | 136 s | 983 m | 11.2 m/s | 1531 (62.5 %) | 11.2 | 169 / 219 ms |
| `live_20261011_145514` | `2026_10_11_14_55_15` | 353 s | 1865 m | 12.8 m/s | 4073 (61.5 %) | 11.5 | 167 / 205 ms |
| `live_20261011_150517` | `2026_10_11_15_05_17` | 254 s | 2342 m | 16.7 m/s | 3030 (60.1 %) | 11.9 | 165 / 206 ms |
| `live_20261011_151614` | `2026_10_11_15_16_15` | 506 s | 3349 m | 17.9 m/s | 6116 (59.7 %) | 12.1 | 164 / 201 ms |

Por frame 78–84 ms (p50): el ritmo de la laptop con batería (§4.20: 75 ms con batería,
50 ms conectada). Antes, a las 14:43, una prueba corta a 5–8 km/h no arrancó el filtro:
solo 3 de 90 fixes a ≥ 3 m/s, y el arranque pide 20 m seguidos a esa velocidad
(`GpsCameraFusion._try_start`); funcionó como está diseñado.

**1. Los tres sensores llegaron completos** (lo recibido en vivo contra lo grabado por
el teléfono, en el tramo común):

| | 145250 | 145514 | 150517 | 151614 |
|---|---|---|---|---|
| Frames grabados / recibidos (no enviados por el teléfono) | 4085 / 4086 (0) | 10575 / 10573 (3) | 7606 / 7596 (11) | 15184 / 15168 (17) |
| Procesados con un frame grabado a ≤ 1 µs | 99.93 % | 99.98 % | 99.97 % | 99.98 % |
| Fixes grabados / recibidos, misma marca y posición | 136 / 136 | 352 / 352 | 254 / 254 | 507 / 507 |
| Muestras IMU grabadas / recibidas, mismos valores | 7204 / 7204 | 18647 / 18647 | 13411 / 13411 | 26771 / 26771 |

El único frame sin par por sesión es el primero, anterior al inicio de la grabación.
Fixes cada 1.0 s sin huecos; IMU cada 18.9 ms (máximo 19 ms); 0 JPEG ni UDP inválidos;
todas las consultas de hora respondidas.

**2. El EKF usó los tres** (`evaluate.py --fusion` sobre cada sesión):

| | 145250 | 145514 | 150517 | 151614 |
|---|---|---|---|---|
| Arranque | 13.7 s | 2.6 s | 14.0 s | 25.4 s |
| Frames girados con el giroscopio | 99.9 % | 100 % | 100 % | 100 % |
| Correcciones de posición (rechazadas) | 122 (0) | 349 (0) | 239 (0) | 481 (0) |
| Velocidad / curso / cámara | 122 / 118 / 116 | 348 / 212 / 254 | 236 / 219 / 214 | 474 / 355 / 301 |
| Alto aceptadas (rechazadas) | 0 | 80 (0) | 5 (0) | 99 (38) |
| Predicción contra cada fix nuevo p50 / p95 | 1.3 / 3.2 m | 1.4 / 3.4 m | 2.0 / 4.3 m | 2.0 / 5.3 m |
| NIS medio de posición (esperado 2) | 0.17 | 0.18 | 0.37 | 0.49 |

La fusión sigue al GPS con ±1.4–3 m. El NIS de posición muy por debajo de 2 dice que la
varianza del GPS en el filtro es conservadora (sin cambiar; anotar para el RL, que
ajusta esos pesos). Los 38 altos rechazados de 151614 son la lluvia (punto 3): la cámara
decía detenido con el carro andando y la compuerta lo rechazó.

**3. Por qué el trazo de la cámara rígida salió mal en 151614: lluvia en el
parabrisas.** El trazo "Cámara con escala propia (rígida)" de
`trajectory_live_metric.png` es la cámara sola (su giro y su escala), alineada al GPS con
una sola rotación y traslación. Comparada con el giroscopio (`evaluate.py`):

| | 145250 | 145514 | 150517 | 151614 |
|---|---|---|---|---|
| Rumbo cámara − giroscopio, final (máximo) | +24° (30°) | +71° (74°) | +106° (123°) | **−452° (470°)** |
| Giro de la cámara / giroscopio en frames que giran > 2° | 1.16* | 1.16 | 1.20 | **−0.04** |
| Cámara detenida con el carro a > 1 m/s | 0.3 % | 0.9 % | 1.1 % | **13.4 % (410 m perdidos)** |
| Distancia cámara / GPS | 1564 / 983 m | 2817 / 1865 m | 2322 / 2342 m | 2708 / 3349 m |
| ATE rígido, mediana | 154 m | 130 m | 230 m | 443 m |

(*145250: solo 30 frames con giro > 2°; el error por frame en ellos es −0.24°.)

- **En 151614 la cámara no ve los giros.** Las ventanas de 3 s con más error son giros
  que la cámara se salta: a los 21 s el giroscopio gira +75° y la cámara −9°; a los
  18 s, +68° contra +10°; a los 3 s, +72° contra +25°; a los 249–252 s, +81° contra
  +14°. Doce ventanas suman −303° de los −452°. En el video hay gotas en el parabrisas
  en esos tramos (el limpiador lo deja limpio a ratos): ORB empareja las gotas, que se
  mueven con la cámara. Medido entre dos frames a 100 ms, con el mismo ORB y máscara:
  a los 21 s el giroscopio gira 4.33° (~66 px de desplazamiento esperado) y los
  emparejamientos se mueven 6 px (p50), giro estimado 0.64°; a los 251 s, 1.80°
  (~27 px) contra 4 px y 1.26°; en un giro limpio a los 181 s, 3.31° contra 68 px y
  3.23°. Las mismas gotas explican las paradas falsas (13.4 %) y los altos rechazados.
  **Es de esa sesión** (la que más llovió); en las otras tres la cámara gira 1.16–1.20
  veces el giroscopio en los giros, la ganancia de siempre (§4.16).
- **Lo general: la cámara sola nunca dio un buen trazo** (§4.12, §4.16), y en campo
  tampoco. En las tres sesiones secas el rumbo deriva +24 a +106° (sobre todo en rectas
  rápidas: en 150517, +12 a +77° en 30 s a 10–16 m/s después de la rotonda) y la escala
  de la cámara es errática: cámara/Doppler 1.2–1.8 en las primeras dos y 0.57 a
  14–18 m/s en 150517; el filtro la ve moverse de 0.4 a 2.5. Con una sola alineación
  rígida, cualquier error de rumbo gira todo lo que sigue. Por eso el rumbo sale del
  giroscopio (decisión del 7/10).
- **El descarte (~60 %) no explica lo de 151614.** Replay determinista de la
  grabación en la laptop (los 15203 frames, 0 % de descarte, mismos intrínsecos y
  máscara del 25 %, con escala; 22 min):

  | 151614 | En vivo (59.7 % descartado) | Replay determinista (0 %) |
  |---|---|---|
  | Giro de la cámara a los 21 s (giroscopio +74–75°) | −9° | −0.8° |
  | Rumbo cámara − giroscopio, final (máximo) | −452° (470°) | −501° (531°) |
  | Cámara detenida a > 1 m/s | 13.4 % (410 m) | 12.1 % (413 m) |
  | Error de giro por frame p50 / p90 (giro real p50) | 0.58 / 1.85° (0.62°) | 0.33 / 1.04° (0.32°) |
  | ATE rígido, mediana | 443 m | 200 m |
  | Cámara sola / cámara + giroscopio, separación final | 70 % / 34 % | 33 % / 34 % |
  | Fusión: predicción contra cada fix p50 / p95 | 2.0 / 5.3 m | 1.8 / 4.1 m |
  | Fusión: altos rechazados | 38 | 41 |

  Las mismas ventanas encabezan el error en los dos (21, 18, 249–252, 285 s): la lluvia
  falla igual con todos los frames. El descarte sí empeora la cámara sola en el resto
  (la separación final de la cámara sola pasa de 33 a 70 %, el ATE de 200 a 443 m),
  pero no la fusión ni la cámara + giroscopio, que no toman el giro de la cámara.

**4. El trazo cámara + giroscopio (sin GPS) ya existe.** Es el modo `visual_only` de
`GpsCameraFusion`: el GPS solo da el punto y el rumbo de arranque; después, giro del
giroscopio y metros de la cámara. Se activa:
- En la evaluación: `evaluate.py <frames.csv> --recording <carpeta> --fusion` dibuja en
  `fusion.png` "Cámara sola" y "Cámara y giroscopio, desde el mismo inicio" junto al GPS
  y la fusión, e imprime su deriva. `--recording` puede ser la grabación del teléfono o
  la propia carpeta de la sesión en vivo (trae `gps.csv` y `gyro_accel.csv`; en 150517
  da 474 m de separación final contra 478 m con la grabación). `--out` para no escribir
  junto al `frames.csv`.
- Como la fusión de la corrida: `run_replay --visual-only` o `run_live --visual-only`
  (reemplaza al EKF completo en esa corrida; la ventana no tiene la opción).
- **No** está en `trajectory_live_metric.png` (ese gráfico solo tiene GPS, cámara rígida
  y fusión).

| Separación al GPS desde el mismo inicio, al final (% de lo recorrido) | 145250 | 145514 | 150517 | 151614 |
|---|---|---|---|---|
| Cámara sola | 516 m (55 %) | 796 m (44 %) | 1010 m (44 %) | 2297 m (70 %) |
| Cámara + giroscopio | 388 m (41 %) | 549 m (30 %) | 478 m (21 %) | 1122 m (34 %) |

Con el giroscopio la forma de los giros sale bien; lo que queda es la escala de la
cámara (el largo de los tramos).

**5. La "rotonda" de 150517: la cámara no la captó mejor.** El lazo que dibuja la cámara
rígida cerca de (0–50, 1050–1150) m son los segundos 140–210 de la cámara. La rotonda
está en los segundos 139–151: el giroscopio marca −20° a la derecha al entrar y +75° a
la izquierda al rodearla (+55° netos, a 3–8 m/s). Rumbo acumulado desde el segundo 125,
contra el giroscopio (referencia):

| | Diferencia p50 / máx (125–185 s) | En la rotonda (139–151 s) |
|---|---|---|
| Curso del GPS (entre fixes) | 5.4° / 8.8° | +4 a +8° |
| Fusión | 1.7° / 3.5° | +0.6 a +3.5° |
| Cámara | 16.4° / 77.6° | −6 a +12° |

Desde la misma posición y rumbo en el segundo 132 (giroscopio + Doppler como
referencia), en 132–156 s: GPS a 2.7 m (mediana; 5.1 m máximo), fusión a 2.0 m (8.1 m),
cámara sola a 5.1 m (45 m), giro de la cámara + Doppler a 3.1 m (30 m). **Las
coordenadas del GPS siguen la rotonda dentro de ~3 m**, y la fusión también (con el
giroscopio). El lazo de la cámara es error suyo: cierra la curva de más (+12° a la
salida) y en la recta rápida que sigue (10–16 m/s) deriva hasta +77° sin girar; además
se salta casi todo el giro a la derecha de 171–177 s (−15° contra −40°). Con la escala de
la cámara a 0.57 en esa recta, el tramo se acorta y la forma parece un lazo. A la escala
del gráfico completo (1.7 km de alto), una rotonda de ~15 m de radio es un quiebre
pequeño en el GPS: se ve suave, pero está.

**6. El "giro de 180°" del GPS en 151614 es real: el carro dio la vuelta a una cuadra y
volvió por la misma calle.** **Confirmado por el estudiante con el video (11/10): los
giros de 183, 196, 250 y 284 s coinciden con lo que se ve; caso cerrado.** Recorrido (GPS y giroscopio coinciden en cada giro):
0–40 s sale del parqueo (+146° a los 17–23 s); 50–80 s al oeste; 85–90 s derecha, al
norte; 90–180 s al norte (alto en 120–137 s); **180–190 s derecha (−83°), al este;
190–200 s izquierda (+88°), al norte; 210–237 s detenido; 250–260 s izquierda (+86°),
al oeste; 280 s alto frente a un muro (intersección en T); 283–292 s izquierda (+86°),
al sur**; 290–506 s de vuelta al sur por la misma calle, pasa junto a la salida (430 s)
y sigue.
- **No es el GPS:** un fix cada 1.00 s sin huecos, velocidad entre fixes contra la
  Doppler 0.22 m/s (p50, máximo 1.8), 0 saltos; el curso del GPS contra el giroscopio,
  0° (p50), −8 a +7° (p5–p95).
- **No es el EKF:** sin cortes de GPS no hubo extrapolación (predicción ≤ 1 s, 0
  correcciones de posición rechazadas); la fusión sigue al GPS.
- **No es el gráfico:** ejes este/norte de UTM 16; los giros del GPS tienen el mismo
  signo que los del giroscopio (con ejes espejados saldrían al revés).
- **Medido sin las posiciones del GPS** (giroscopio + Doppler): la vuelta a la cuadra
  (170–290 s) deja al carro 33 m al este de la calle de ida (el GPS: 15 m); el tramo al
  este mide +150 m y el del oeste −142 m (GPS: +148 y −146). Una calle paralela estaría
  a una cuadra (~100 m). A la ida y a la vuelta, en y = 175, 420 y 600 m, el GPS separa
  las dos pasadas 7 m.
- **En el video:** giro a la derecha a los 183–187 s (hacia el muro con murales), a la
  izquierda a los 196 s y a los 250–256 s, alto frente a un muro con semáforo a los
  280 s y giro a la izquierda a los 284–292 s.

**7. Todas las sesiones son coherentes** (incluida la prueba lenta de las 14:43):

| Sesión | Coordenadas | Fixes: intervalo máx / saltos | Curso GPS − giroscopio p5–p95 (máx) | Giro neto en marcha |
|---|---|---|---|---|
| 144332 | 9.8801–9.8804, −84.0646–−84.0641 | 1.00 s / 0 | ±2° (2°), solo 3 pares > 3 m/s | −2° |
| 145250 | 9.8856–9.8914, −84.0721–−84.0664 | 1.00 s / 0 | −3.4 a +2.5° (8°) | −174° |
| 145514 | 9.8917–9.9025, −84.0769–−84.0704 | 1.00 s / 0 | −5.8 a +4.8° (10°) | +93° |
| 150517 | 9.9025–9.9182, −84.0760–−84.0722 | 1.00 s / 0 | −4.3 a +5.3° (11°) | −255° |
| 151614 | 9.9100–9.9258, −84.0754–−84.0721 | 1.00 s / 0 | −8.1 a +7.4° (27°, a los 357–358 s) | +144° |

Todas en Costa Rica (Valle Central, altitud 1145–1205 m), zona UTM 16 (EPSG 32616,
la correcta al oeste de −84°; el límite con la 17 está a ~7 km al este), la misma en la
sesión en vivo. Velocidad entre fixes contra Doppler ≤ 1.8 m/s en todas. Las sesiones
se encadenan: 145250 → 145514 a 56 m en 8 s (en marcha), 145514 → 150517 a 4 m, 150517
→ 151614 a 14 m (detenido entre medio). El video gira con el giroscopio (en los frames
limpios, la cámara 1.16–1.20 veces en los giros, punto 3), y en el giro más grande de
cada sesión los frames antes y después muestran el giro.

**Resumen de confiabilidad:**
- **GPS, IMU y fusión: confiables en las cinco sesiones.** Sin huecos, sin saltos,
  coordenadas y zona correctas, y el curso del GPS coincide con el giroscopio.
- **Cámara sola (trazo rígido): no confiable en ninguna**, por la deriva de rumbo y la
  escala errática (punto 3). En 151614 además falla por la lluvia (no ve los giros, se
  cree detenida); 150517 tiene gotas en partes.
- **Escala de la cámara: no confiable en campo** (cámara/Doppler de 0.6 a 1.8 según la
  sesión y la velocidad).
- 144332 sirve solo para verificar los sensores: no arrancó el filtro (5–8 km/h).

**8. El gráfico de cada sesión, ampliado (11/10/2026).** `plot_trajectory`
(`run_replay.py`), que hace `trajectory_<modo>_metric.png` al terminar una sesión en
vivo o un replay con `--trajectory`:
- **Sentido y tiempo:** flechas a lo largo de cada trazo (`direction_arrows`, 12 en el
  GPS y 8 en los demás, repartidas por distancia), la marca m:ss sobre el GPS cada
  minuto (cada 30 s si la sesión dura menos de 3 min; `time_marks`), Inicio (círculo)
  y Fin (cuadrado). Ejes "Este"/"Norte". La leyenda va debajo del mapa para no tapar
  el recorrido. Con eso se distingue la ida de la vuelta por la misma calle (151614).
- **El cuarto trazo, "Cámara + giroscopio, sin GPS desde el arranque"**
  (`camera_gyro_track`): `run_fusion(..., visual_only=True)` sobre los frames de la
  sesión, el mismo que dibuja `evaluate.py --fusion`. Para eso `OnlineFusion` guarda
  en `fixes` cada fix que le entrega al filtro (tiempo, posición, Doppler) y
  `session.py` y `run_replay.py` se los pasan a `plot_trajectory`. Solo con escala
  métrica y giroscopio. El panel de la derecha suma su separación al GPS y la consola
  imprime la mediana y la final. La cámara sola sigue con su alineación rígida (sin
  cambios: mismas medianas, 159.6 / 152.7 / 233.6 / 446.5 m).
- Verificado: rehecho el gráfico de las cuatro sesiones desde `frames.csv` y
  `gps.csv`: separación final de cámara + giroscopio 41 / 30 / 21 / 34 %, lo mismo
  que `evaluate.py --fusion`. Replay de 1500 frames de `14_52_51` con `--trajectory`,
  y sesión en vivo de 45 s con el simulador sirviendo esa grabación (`run_live`, sin
  escala: el gráfico no métrico no lleva el cuarto trazo). Invariantes 6/6.

**9. El trazo cámara + giroscopio de 151614 cubre toda la sesión, pero no llega al
final.** Arranca a los 25.4 s (cuando arranca la fusión: es el mismo arranque) y sigue
hasta el último frame (506.1 s); no se congela. En el gráfico parece cubrir solo el
inicio porque:
- **Se pasa al norte en el alto de 209–237 s.** El modo solo visual tiene la velocidad
  como estado: llegó al alto con v = 15 m/s y rechazó los altos de la cámara 18 s
  seguidos (diferencia demasiado grande para la compuerta, y sin GPS que la
  corrija). Avanzó 600 m en 200–240 s con el carro casi quieto (GPS: 82 m) y, al
  aceptar un alto a los 227.6 s con la incertidumbre de posición en 281 m, retrocedió
  157 m en un frame (el salto del panel de la derecha, keyframe ~2770). En las demás
  sesiones: 145250 y 150517 sin saltos > 20 m; 145514 tres de 21–33 m (66, 154 y
  165 s); 151614 otro de 33 m a los 404 s. Es del modo solo visual: la fusión con GPS
  no lo tiene. Sin cambiar (decisión pendiente del estudiante).
- **Se queda corto de vuelta.** Desde los 240 s avanza con los metros de la cámara,
  y con la lluvia esos metros son la mitad: 1128 m del trazo verde (cámara 1033 m)
  contra 2035 m del GPS (Doppler) entre 240 y 506 s; los peores tramos, 240–260 s
  (9 m contra 165 m), 300–320 s (27 contra 123) y 440–480 s (142 contra 599). El rumbo
  del giroscopio es correcto (al sur desde los 280 s), así que vuelve sobre su propio
  trazo de subida y termina en (−319, 261) m, a mitad de la calle de ida; el carro
  terminó en (−210, −850).

**10. Diseño de los trazos (decidido por el estudiante el 11/10/2026).**
- **En tiempo real (la ventana): dos líneas, la fusión (trazo oficial) y el GPS (la
  referencia).** Hasta el 11/10 la ventana no tenía mapa (era el paso 3 del plan); ahora
  lo tiene, debajo del video (`TrackMap` en `live_window.py`, 640 × 300): metros al
  este y al norte del primer fix con la misma escala en los dos ejes, ajustado a lo
  recorrido (mínimo 60 m), barra de escala, norte, punto de inicio y la posición
  actual de cada uno; sin fondo de calles (decisión del 5/10). Se actualiza dos veces
  por segundo con `PipelineSession.track_since` (los fixes que recibió la fusión y su
  estado por frame; la fusión se dibuja cada 0.5 m). Probado en replay con la ventana
  real (15_05_17 desde el segundo 130, 45 s): 32 fixes y 177 puntos de fusión; el punto
  de la fusión va adelante del de GPS, que llega hasta 1 s viejo.
- **Al terminar (el análisis, `trajectory_<modo>_metric.png`): las cuatro líneas** y el
  gráfico de desviación (punto 8). La leyenda dice qué es cada una: "GPS del teléfono
  (referencia)", "Cámara sola (diagnóstico; sin GPS, deriva)", "Cámara + giroscopio
  (diagnóstico; sin GPS, deriva)" y "Fusión (trazo oficial)".
- El trazo cámara + giroscopio queda como diagnóstico con su limitación documentada
  (punto 9: se adelanta en los altos cuando la compuerta rechaza los de la cámara; con
  lluvia se queda corto).

### 4.22 Robustez de la fusión con el GPS degradado (11/10/2026)

Pregunta: cuando el GPS es malo (ruidoso, con saltos o con pérdidas, no ausente), ¿cuánto
empeora la fusión, le da menos peso al GPS, y compensan la cámara y el giroscopio?
Banco temporal (fuera del repo): `run_fusion` sobre los frames de las sesiones en vivo
150517 (sin lluvia, 254 fixes, hasta 16.7 m/s) y 145514 (353 fixes), con el GPS de la
grabación degradado; la Doppler queda intacta salvo en los fixes perdidos. **La
referencia es el GPS bueno** (no hay ground truth). Error en cada instante de fix
después del arranque, media de 5 semillas. "GPS solo" es el fix degradado, o, si se
perdió, el último entregado llevado a su Doppler en la dirección de los dos últimos.

**150517** (p50 / p95 / máximo, m):

| GPS | GPS solo | Fusión | Fusión sin giroscopio | Posición ok / rechazada / forzada |
|---|---|---|---|---|
| Bueno (referencia) | 0 | 1.8 / 4.1 / 6 | 1.9 / 4.5 / 6 | 239 / 0 / 0 |
| Ruido σ 5 m | 5.8 / 11.8 / 17 | **7.1 / 35.3 / 77** | 7.6 / 30.8 / 62 | 192 / 36 / 10 |
| Ruido σ 10 m | 11.5 / 23.7 / 34 | **14.5 / 47.3 / 100** | 15.0 / 51.2 / 92 | 135 / 90 / 16 |
| Ruido σ 20 m | 22.9 / 47.3 / 68 | 26.0 / 64.9 / 115 | 26.6 / 70.8 / 107 | 99 / 128 / 15 |
| Multitrayecto (correlado) σ 10 m, τ 10 s | 11.9 / 23.0 / 29 | 12.7 / 26.3 / 50 | 11.9 / 25.7 / 32 | 214 / 19 / 6 |
| Saltos sueltos de 30–100 m, 5 % de los fixes | 0 / 17.0 / 95 | **2.1 / 7.4 / 57** | 2.0 / 5.9 / 47 | 226 / 14 / 0 |
| Saltos sueltos, 15 % | 0 / 76.9 / 99 | **2.8 / 35.3 / 86** | 2.4 / 23.8 / 80 | 192 / 45 / 3 |
| Ráfagas de 5 s a 30–100 m (3 % de inicio) | 0 / 66.4 / 83 | 2.3 / 51.5 / 79 | 2.6 / 51.8 / 79 | 202 / 25 / 12 |
| Pérdida del 50 % de los fixes | 0.1 / 9.1 / 30 | 2.3 / 10.9 / 32 | 2.5 / 12.6 / 36 | 115 / 1 / 0 |
| Pérdida del 80 % | 3.0 / 77.0 / 142 | 8.0 / 100.9 / 138 | 10.4 / 128.7 / 195 | 35 / 6 / 2 |
| Un fix cada 5 s | 1.5 / 16.2 / 33 | 4.5 / 50.4 / 85 | 5.1 / 50.5 / 86 | 40 / 5 / 2 |
| Un fix cada 10 s | 6.3 / 55.1 / 79 | 18.4 / 190.5 / 202 | 20.6 / 195.9 / 231 | 13 / 7 / 3 |
| Combinado: σ 10 m + saltos 5 % + pérdida 50 % | 14.4 / 52.4 / 100 | 20.1 / 117.7 / 222 | 20.8 / 99.7 / 206 | 44 / 53 / 16 |

**145514**, lo distinto: ruido σ 5 m 5.8 / 29.1 (GPS solo 5.8 / 12.0); saltos 5 %
1.6 / 12.1 (GPS 0 / 15.1); ráfagas 2.0 / 52.3 (GPS 0 / 65.4); **pérdida del 80 % 3.6 /
25.6 (GPS 1.6 / 42.6) y un fix cada 5 s 2.7 / 12.1 (GPS 0.9 / 12.1)**: ahí la cámara y
el giroscopio sí cubren los huecos.

**Lo que aguanta:**
- **Saltos sueltos:** la compuerta los rechaza. Con 5 % de fixes saltados el p95 queda
  en 7–12 m contra 15–17 del GPS solo; con 15 %, 24–35 contra 75–77.
- **Pérdidas moderadas** (50 %): igual que con todos los fixes (p95 6.5–10.9 m).
- **Pérdidas grandes, si la escala de la cámara está bien** (145514): mejor que el GPS
  solo.

**Lo que no aguanta, y por qué** (variantes medidas en el banco, sin cambiar el código;
p50 / p95 en m, 150517 y 145514):

| | Como está | Sin el curso del GPS | Varianza del GPS ajustada al ruido | Sin la velocidad como estado |
|---|---|---|---|---|
| Ruido σ 5 m | 7.1 / 35.3 · 5.8 / 29.1 | **3.0 / 6.1 · 2.2 / 4.9** | **3.4 / 6.3 · 2.3 / 5.1** | 7.8 / 34.7 · 6.3 / 29.4 |
| Ruido σ 10 m | 14.5 / 47.3 · 11.6 / 40.4 | 6.4 / 15.4 · 4.9 / 11.6 | **5.1 / 11.0 · 3.2 / 10.1** | 14.5 / 46.2 · 11.9 / 39.9 |
| Multitrayecto σ 10 m | 12.7 / 26.3 · 10.9 / 24.1 | 11.0 / 21.3 · 9.6 / 20.7 | 10.6 / 21.2 · 8.5 / 19.0 | 13.3 / 28.2 · 11.8 / 25.2 |
| Ráfagas de 5 s | 2.3 / 51.5 · 2.0 / 52.3 | 3.1 / 57.5 · 2.3 / 50.8 | igual | 2.5 / 58.1 · 2.0 / 54.8 |
| Un fix cada 5 s | 4.5 / 50.4 · 2.7 / 12.1 | igual | igual | 4.7 / 52.5 · 2.9 / 17.3 |

- **El EKF no le da menos peso al GPS degradado.** La posición entra siempre con
  σ = 4 m (`GPS_SIGMA_M`) y el curso entre fixes con σ = 0.5 m (`COURSE_SIGMA_M`); su
  única defensa es la compuerta (y forzar tras dos rechazos). Con ruido, el NIS medio
  de posición sube de 0.37 a 3.8–5.0 y rechaza hasta la mitad de los fixes, pero los
  cursos ruidosos que pasan la compuerta tuercen el rumbo y la predicción se va: con
  σ 5–10 m la fusión queda **peor** que el GPS solo (p95 29–47 contra 12–24 m).
- **Si el peso fuera el correcto, compensa:** con la varianza ajustada al ruido real,
  la fusión filtra el GPS a la mitad (σ 5 m: p95 5–6 m contra 12; σ 10 m: 10–11 contra
  24). Sin el curso pasa casi lo mismo: el curso es el que hace daño con ruido. Es
  exactamente lo que el RL tiene que aprender (darle más o menos peso a un sensor según
  la situación, OE3): aquí está medido cuánto se gana.
- **Error correlado (multitrayecto):** ninguna variante lo quita (p95 19–26 m contra
  23 del GPS solo). Un error lento y coherente parece movimiento; sin otra referencia
  absoluta no se distingue.
- **Ráfagas de varios segundos:** tras dos rechazos el filtro fuerza el tercer fix
  si el carro pudo llegar ahí (`MAX_REJECTED`, `V_MAX`), y sigue el salto (p95 ~52 m).
  El diseño elige no quedarse ciego si el error era de la predicción; el costo es este.
- **Fixes escasos:** el error es a lo largo del camino (un fix cada 5 s en 150517: p95
  50 m a lo largo, 18 m de costado). Entre fixes la velocidad del filtro va de 0.38 a
  1.81 veces la Doppler: la escala de la cámara, errática en campo (§4.21). El
  giroscopio sostiene el rumbo; los metros de la cámara no.
- **El giroscopio casi no cambia estos números** (con GPS, el curso y la posición ya
  corrigen el rumbo); su aporte está en los cortes y sin GPS (§4.17, §4.21 punto 4).

**Para la tesis:** la fusión es robusta a saltos sueltos y pérdidas moderadas, pero no a
GPS ruidoso con los pesos fijos; con el peso del GPS adaptado al ruido sí lo es (mide la
mitad del error del GPS solo). Cambios posibles, ninguno hecho (decisión del
estudiante): (a) que el curso entre fixes exija que la cuerda coincida con la Doppler
por el tiempo transcurrido, o subir su σ; (b) adaptar `GPS_SIGMA_M` con el NIS reciente
(o dejarlo al RL); (c) no forzar un fix que llega tras una ráfaga hasta que dos
seguidos coincidan entre sí.


### 4.23 Curso condicionado a la Doppler y ráfagas del GPS (12/10/2026)

Decisiones del estudiante sobre §4.22: (a) sí, (b) no (el peso del GPS es el OE3, queda
para el RL), (c) sí. Implementado en `GpsCameraFusion` (`fusion.py`):

- **(a) El curso entre dos fixes solo se usa si la cuerda coincide con la Doppler:**
  |cuerda − v̄·Δt| ≤ `COURSE_DOPPLER_M` = 1.5 m. En las cuatro grabaciones de campo
  limpias la diferencia es 0.24 m (p50), 1.17 m (p99) y 1.8 m como máximo: se pierde 1
  curso de 355 en 151614 y ninguno en las demás. Con ruido σ 5 m pasa ~20 % de las
  cuerdas; con σ 10 m, ~10 %.
  **Más, decidido al medir (no estaba en la propuesta): con el giroscopio girando el
  rumbo, el curso nunca se fuerza** (`may_force=not self._from_gyro`); sin
  giroscopio se fuerza como antes, porque ahí es lo único que recupera el rumbo de la
  cámara. El largo de la cuerda no ve el error de costado de un fix ruidoso: con solo
  la condición de la Doppler, el ruido σ 5 m quedaba en p95 25.7 m (antes 35.3); con
  el curso sin forzar, 6.5 m.
- **(c) Ráfagas: no se fuerza un fix mientras el GPS siga desplazado por el mismo
  vector.** Una ráfaga empieza con un fix cuya cuerda falla la Doppler por más de
  `GPS_JUMP_M` = 10 m (nunca con fixes limpios; máximo 1.8 m) y que se aleja de una
  predicción con la que el fix anterior coincidía. Mientras los fixes rechazados
  queden desplazados por el mismo vector (± `BURST_MATCH_M` = 8 m), ninguno se fuerza,
  hasta `JUMP_HOLD_S` = 10 s desde el primer salto; un nuevo salto dentro de la ráfaga
  actualiza el vector. El forzado tras un corte de GPS no cambia (no hay salto entre
  fixes seguidos).
  **La regla literal ("forzar cuando dos seguidos coinciden") no sirve:** los fixes de
  una ráfaga se mueven juntos, así que dos seguidos coinciden. Y la primera versión
  (no forzar nada durante 10 s tras cualquier salto de la cuerda) rompía el GPS
  ruidoso: con σ 10 m el ~45 % de las cuerdas pasa de 10 m, el filtro no podía forzar
  nunca y se iba (p95 793 m). Por eso la condición es el mismo vector de
  desplazamiento, que el ruido no repite.
- Contadores nuevos en `metrics.json` → `fusion`: `saltos_del_gps`,
  `forzados_retenidos_tras_salto`, `cursos_descartados_por_doppler`.

**Antes y después** (banco de §4.22, 5 semillas, contra el GPS bueno; p95 en m;
"sin gyro" es la fusión con el rumbo de la cámara):

| GPS | 150517 antes → después | 145514 antes → después | GPS solo | Sin gyro, después (150517 · 145514) |
|---|---|---|---|---|
| Bueno | 4.1 → 4.1 | 3.1 → 3.1 | — | 4.5 · 3.1 |
| Ruido σ 5 m | 35.3 → **6.5** | 29.1 → **5.3** | 11.8 · 12.0 | 21.9 · 21.4 |
| Ruido σ 10 m | 47.3 → **15.8** | 40.4 → **11.6** | 23.7 · 24.0 | 35.4 · 31.3 |
| Ruido σ 20 m | 64.9 → **39.7** | 55.7 → **36.8** | 47.3 · 48.0 | 60.9 · 51.4 |
| Multitrayecto σ 10 m, τ 10 s | 26.3 → 21.0 | 24.1 → 20.4 | 23.0 · 23.1 | 24.0 · 23.8 |
| Saltos sueltos 5 % | 7.4 → 7.9 | 12.1 → **6.1** | 17.0 · 15.1 | 5.1 · 3.2 |
| Saltos sueltos 15 % | 35.3 → **4.7** | 24.2 → **6.4** | 76.9 · 74.9 | 6.3 · 3.5 |
| Ráfagas de 5 s a 30–100 m | 51.5 → **4.4** | 52.3 → **3.7** | 66.4 · 65.4 | 5.0 · 3.8 |
| Pérdida 50 % / 80 % | 10.9 / 100.9 → igual | 6.5 / 25.6 → igual | 9.1 / 77.0 · 6.2 / 42.6 | 12.6 / 128.7 · 6.8 / 26.9 |
| Un fix cada 5 s / 10 s | 50.4 / 190.5 → igual | 12.1 / 38.6 → igual | 16.2 / 55.1 · 12.1 / 40.3 | 50.5 / 195.9 · 12.4 / 44.1 |
| Combinado | 117.7 → **27.8** | 88.4 → **19.2** | 52.4 · 55.6 | 46.1 · 50.9 |

- **Con GPS bueno nada cambia** (mismas correcciones aceptadas en las cuatro sesiones;
  predicción contra cada fix 1.3–2.0 m p50 como en §4.21). Barrido de cortes de 10–120
  s: idéntico al de antes en 150517 y 145514. Invariantes 6/6.
- **Ahora la fusión filtra el GPS ruidoso** (σ 5 m: p95 5–6.5 m contra 12 del GPS
  solo; σ 10 m: 12–16 contra 24) y **ya no sigue las ráfagas** (4 m contra 52).
- Lo que no cambió: el multitrayecto (error lento y coherente: parece movimiento) y los
  fixes escasos (escala de la cámara, §4.22).
- **Sin giroscopio la mejora con ruido es menor** (σ 5 m: p95 21–22 m): ahí el curso se
  sigue forzando, como corresponde sin otra fuente de rumbo.

**Arreglo de paso:** `evaluate.py --sweep` fallaba en sesiones cortas (`summarize_sweep`
con una duración sin cortes: 145250 dura 136 s y no le cabe uno de 120 s). Ahora omite
esa duración.

### 4.24 Mapa en replay y caché de resultados (12/10/2026)

- **El mapa de la ventana también funciona en replay**: la ventana usa la misma
  `PipelineSession.track_since` en vivo y con una grabación; probado con la ventana real
  sobre 15_05_17 (§4.21, punto 10). El replay guarda frames.csv, gps.csv, metrics.json
  y la trayectoria con las cuatro líneas, como en vivo; no guarda gyro_accel.csv (la IMU
  ya está en la grabación) ni reloj.csv (en replay no hay consulta de hora).
- **Caché de replays deterministas** (`realtime/cache.py`, usada por `run_replay.py`):
  `resultados/cache/<grabación>/<clave>/` con frames, métricas, trayectoria y
  `cache.json` (qué entró en la clave, cuándo y en qué equipo, versiones de OpenCV,
  numpy y torch). La clave hashea (1) el contenido de la grabación (video, tiempos, GPS
  e IMU; ~1 s para 430 MB), (2) las opciones que cambian el resultado y (3) el código
  que lo produce (`pipeline.py`, `depth_scale.py`, `fusion.py`, `sources.py`, más
  `CODE_VERSION` para el bucle de `run_replay.py`). Si algo cambia, no acierta y
  reprocesa; `--no-cache` fuerza. `python -m LMS.LMS_RL_ORB_GPS.realtime.cache` la
  lista. **No cachea** el modo estricto ni la ventana (no son reproducibles). Va en git
  (`.gitignore`: `!resultados/cache/`). Probado: 600 frames procesados en 56 s y
  restaurados en 0.5 s con el JSON idéntico; otra máscara crea otra entrada.
  **Ojo:** el modelo de profundidad en otra GPU o con otro torch puede dar resultados
  algo distintos; `cache.json` dice dónde se procesó cada entrada.
- **Desde el 13/10 cada entrada guarda también el `location.csv` de su grabación**
  (56 KB): con él y el `frames.csv` (que trae `gyro_yaw`) la fusión se vuelve a correr
  sin la grabación, que es sobre lo que entrena el RL. Verificado en 14:05, 14:18 y
  2025_03_11: la misma posición a 0.0000 m y el mismo giro a 5·10⁻¹¹ rad que con la
  grabación entera. Las entradas guardadas antes se completan con
  `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.cache --add-location`, que busca la
  grabación del mismo nombre en `mobile_data/` y **solo copia si el hash de su contenido
  es el de la entrada** (probado con un `location.csv` alterado: no lo copia). El listado
  (`python -m ...realtime.cache`) marca las entradas `(sin location.csv)`. La clave no
  cambió (`cache.py` no está en el código que se hashea): las entradas existentes
  siguen valiendo.


### 4.25 Diagnóstico general del sistema (12/10/2026)

Con el código del 12/10 ((a) y (c) de §4.23). Fuentes: las cuatro sesiones de campo en
vivo (§4.21), sus replays deterministas completos en la laptop (conectada, sin otra
carga salvo en 145250; en la caché, §4.24), los barridos de cortes de GPS, el banco de
GPS degradado (§4.22–4.23) y lo medido antes en la de escritorio (§4.3, §4.16, §4.20).
**No hay ground truth:** la referencia es el GPS del teléfono (al aire libre, sin saltos
en las cuatro sesiones, curso igual al del giroscopio ±3–8°, §4.21).

#### 1. Rendimiento

| | Escritorio (Ryzen 5600X, RTX 3070) | Laptop conectada (i5-8300H, GTX 1050 Ti) | Laptop con batería |
|---|---|---|---|
| Por frame p50 (ORB / emparejamiento / pose) | 23–27 ms (9 / 6 / 7) | 48–54 ms (19–23 / 14–16 / 14) | 75–84 ms (32 / 22 / 20) |
| Frecuencia efectiva en vivo | 30 Hz | ~19 Hz | 11–13 Hz |
| Descarte en vivo | 0.1–1 % | 37 % | 56–62 % |
| Captura → resultado p50/p95, por cable | 97 / 107 ms: cumple | 132–138 / 167–189 ms: **no cumple** | 164–169 / 201–219 ms: **no cumple** |
| Modelo de profundidad por estimación | 88–97 ms | 291 ms (hilo, en vivo); 423–427 ms (en línea, replay) | 442 ms |
| EKF por frame p50 / p95 | 0.14 ms | 0.15–0.23 / 0.36–0.46 ms | 0.36–0.40 / 0.93–1.04 ms |

- **30 Hz con la fusión: sí en la de escritorio; no en la laptop** (frecuencia ≥ 10 Hz
  sí, p95 < 150 ms no). La fusión no es el cuello: 0.2–0.4 ms de 50–84. El cuello es el
  front-end ORB (fijo por el invariante T6, §4.20).
- **Costos:** EKF 0.2 ms por frame (incluye integrar el giroscopio; la IMU llega a 53 Hz
  por el socket del GPS, sin costo medible aparte). Escala: +2.3 ms de latencia en la de
  escritorio (§4.3); en la laptop el p95 en la PC sube de 92 a 100 ms (§4.20).
- **En vivo y en replay, la fusión da lo mismo:** predicción contra cada fix p50 / p95
  1.3 / 3.2 contra 1.2 / 3.1 m (145250), 2.0 / 4.3 contra 1.8 / 4.0 (150517), 1.4 /
  3.4 contra 1.3 / 3.0 (145514), 2.0 / 5.3 contra 1.8 / 4.1 (151614). El replay
  determinista reproduce `run_fusion` a 0.5 mm (§4.19). **La cámara no da lo mismo:**
  procesando todos los frames su rumbo deriva **más** (+48 contra +24°, +194 contra
  +106°, +140 contra +71°): entre frames más separados gira con menos deriva (§4.16),
  y el descarte de la laptop hace de esa separación.

#### 2. Odometría visual

| Replay determinista (30 Hz) | 145250 | 145514 | 150517 | 151614 (lluvia) |
|---|---|---|---|---|
| Error de giro por frame p50 / p90 (giro real p50) | 0.33 / 0.83° (0.38°) | 0.29 / 0.90° (0.32°) | 0.37 / 0.97° (0.39°) | 0.33 / 1.04° (0.32°) |
| Rumbo cámara − giroscopio al final (máximo) | +48° (68°) | +140° (142°) | +194° (200°) | −502° (532°) |
| Distancia cámara / GPS | 1605 / 987 m | 2814 / 1867 m | 2267 / 2342 m | 2662 / 3351 m |
| Cámara/Doppler por banda (p50) | 1.47–1.83 | 1.3–1.8 (en vivo) | 1.28 a 10–12 m/s; **0.54–0.58 a 12–18** | **0.16** a 14–18 m/s |
| Cámara "detenida" con el carro a > 1 m/s | 0.0 % | 0.7 % | 0.6 % | **11.4 % (400 m)** |
| Cámara sola / cámara + giroscopio, separación final | 73 % / 39 % | 69 % / 31 % | 97 % / 21 % | 33 % / 34 % |

- **La VO no sirve como trazo por sí sola.** El error de giro por frame es del tamaño
  del giro típico; en los giros (> 2°) la cámara gira 1.16–1.20 veces el giroscopio, y
  en recta suma un sesgo: +48 a +194° de deriva en 2–8 min (~10–25°/min; en la autopista
  de la de escritorio, +55°/min, §4.16). **La escala es errática:** de 0.16 a 1.8 veces
  la Doppler según la sesión y la velocidad; a más de 12 m/s colapsa.
- **Dónde falla:** lluvia en el parabrisas (no ve los giros: −0.04 veces el giroscopio;
  se cree detenida en el 11–13 % de la marcha, §4.21); velocidad alta (escala 0.16–0.6
  a más de 12 m/s); poca textura (bajo techo sin avanzar, inliers p50 16–21 y 15–22
  poses descartadas por giro imposible, §4.20). **No medido en campo:** noche, túneles,
  autopista con el código nuevo (lo más rápido fue 17.9 m/s).
- **Lo que sí aporta:** la velocidad entre fixes (1 s) y los altos (la compuerta ve el
  84–93 % de los altos del GPS, con 0–1.1 % de altos falsos en seco).
- **El giroscopio:** baja la separación de la cámara sola de 33–97 % a 21–39 %, y
  sostiene el rumbo de la fusión: en cortes de 120 s de 151614, p50 413 → 187 m; de 60 s
  en 145250, 108 → 71 m (barridos, filas "cámara" contra "giroscopio").

#### 3. El filtro

**a) Con los sensores bien** (cuatro sesiones, en vivo):
- **Arranca** cuando el GPS ve 20 m seguidos a ≥ 3 m/s: a los 2.6, 13.7, 14.0 y 25.4 s
  (este último, salida lenta de un parqueo). A 5–8 km/h no arranca (14:43).
- **Sigue al GPS:** predicción de 1 s contra cada fix nuevo p50 1.3–2.0 m, p95 3.2–5.3
  m; 0 correcciones de posición rechazadas; giroscopio en el 99.9–100 % de los frames.
- **El NIS de posición es 0.17–0.49** (esperado 2): la varianza del GPS (4 m) es
  conservadora con este GPS. No se toca: es el peso que le toca al RL.

**b) Cuando un sensor falla** (p50 / p90 o p95 en m, según se indica):

| Falla | Cómo se comporta | Error |
|---|---|---|
| **GPS cortado** 10 / 20 / 30 / 60 / 120 s (al volver, p50 / p90; 3–4 sesiones) | Sigue con giroscopio + velocidad (estado) + cámara; le gana al GPS a velocidad constante en el 55–100 % de los cortes | 10–22 / 20–61; 17–54 / 62–163; 34–161 / 80–255; 71–242 / 144–597; 156–434 / 427–680 |
| **GPS ruidoso** σ 5 / 10 / 20 m (p95) | Filtra el ruido (curso solo si cuadra con la Doppler, §4.23) | 5.3–6.5 / 11.6–15.8 / 36.8–39.7 (GPS solo 12 / 24 / 47) |
| **GPS con saltos** sueltos 5–15 % / ráfagas de 5 s (p95) | Los rechaza; no sigue las ráfagas | 4.7–7.9 / 3.7–4.4 (GPS solo 15–77 / 65) |
| **GPS con multitrayecto** (error correlado, σ 10 m, p95) | No lo distingue de movimiento | 20–21 (GPS solo 23) |
| **GPS escaso** (1 fix cada 5 / 10 s, p95) | Depende de la escala de la cámara | 12–50 / 39–191 (GPS solo extrapolado 12–16 / 40–55) |
| **Cámara con lluvia** (151614) | El rumbo no se afecta (giroscopio); la compuerta de altos rechaza 38–42 altos falsos | predicción 2.0 / 5.3 en vivo (seco 1.3–2.0 / 3.2–4.3); cortes de 30 s 74 / 186 |
| **Sin IMU** (rumbo de la cámara) | Con GPS bueno, igual; con GPS malo o cortado, peor | GPS bueno p95 3.1–4.5; ruido σ 5 m p95 21–22 (con IMU 5–6.5); cortes de 120 s en 151614 p50 413 (con IMU 187) |
| **Dos a la vez: GPS cortado + lluvia** (151614) | El giroscopio sostiene el rumbo; la escala de la cámara empeora | 10 / 30 / 60 s: 17 / 74 / 150 p50 (seco 10–22 / 34–161 / 71–242) |
| **GPS cortado + sin IMU** | Rumbo de la cámara: deriva | 151614: 30 s 92 / 230, 60 s 167 / 441 (con IMU 74 / 186, 150 / 323) |
| **GPS ruidoso + sin IMU** (σ 10 m, p95) | El curso se fuerza (sin otra fuente de rumbo) | 31–35 (con IMU 12–16) |
| **GPS combinado (ruido + saltos + pérdidas) + sin IMU** (p95) | | 46–51 (con IMU 19–28) |
| **Solo cámara + giroscopio, sin GPS desde el arranque** | Rumbo bien, largo mal | separación final 21–41 % de lo recorrido |

**c) Qué hace el EKF fijo y qué le queda al RL**

**Lo hace el EKF fijo (medido):** arranca solo; sigue al GPS a 1.3–2.0 m (p50);
rechaza saltos sueltos y ráfagas (p95 4–8 m contra 15–77 del GPS); filtra ruido blanco
a la mitad (p95 5–16 m contra 12–24); el rumbo sale del giroscopio (99.9–100 % de los
frames) y sobrevive a la lluvia; cubre cortes de 10 s con 10–22 m (p50) y le gana al GPS
extrapolado en el 55–100 % de los cortes.

**No lo hace el EKF fijo, y cuánto ganaría una política (oráculo, medido):**

1. **Cuánto confiar en la velocidad de la cámara durante un corte de GPS.** Hoy siempre
   la usa. Ignorarla (Doppler constante + giroscopio) es mejor en el 11–70 % de los
   cortes según la sesión (en 150517, rápida, el 67–70 % de los de 20–30 s). 324 cortes
   de 10–30 s en las cuatro sesiones, error al volver el GPS (p50 / p90 m):

   | Política | 10 s | 20 s | 30 s |
   |---|---|---|---|
   | EKF fijo (siempre con la cámara) | 14.3 / 38.4 | 32.3 / 101.0 | 61.8 / 195.3 |
   | Siempre sin la cámara | 18.4 / 49.5 | 44.2 / 121.7 | 81.7 / 207.1 |
   | Regla de un umbral sobre la velocidad (validación dejando una sesión afuera) | 14.7 / 38.3 | 32.0 / 95.1 | 61.4 / 177.7 |
   | Reglas sobre la razón Doppler/cámara o los inliers (ídem) | 14.7–17.0 / 41.6–49.3 | 30.8–45.3 / 106.6–125.6 | 67.7–81.0 / 189.0–201.7 |
   | **Oráculo (el mejor en cada corte)** | **8.3 / 25.6** | **21.3 / 67.0** | **39.6 / 111.0** |

   **El margen existe y es grande** (30 s: p90 195 → 111 m, −43 %), **pero ninguna regla
   de una señal lo captura**: la mejor (velocidad) recupera el 21 % del p90 a 30 s y nada
   del p50. Correlaciones de las señales con "conviene ignorarla": velocidad +0.32,
   fracción detenida −0.17, inliers +0.10, inestabilidad de la razón +0.10, sesgo +0.05.
2. **El peso del GPS con ruido fuerte.** Después de §4.23 el margen es chico: σ 5 m,
   ninguno (p95 6.5 contra 6.7 del oráculo); σ 10 m, 1–2 m de p95 (15.8 → 14.6, 11.6 →
   9.5); σ 20 m, grande en la mediana (19.9 → 7.8 y 14.2 → 5.7 m) pero no en el p95 (el
   oráculo empeora una sesión, 39.7 → 49.7). **Una regla de una línea (varianza ×
   NIS medio reciente / 2) llega a lo mismo** (σ 20 m: p50 8.2 y 5.3 m), con dos
   retrocesos en 145514 (σ 10 m p95 11.6 → 20.9; combinado 19.2 → 30.9).
3. **Multitrayecto:** ningún peso lo corrige (con el peso ideal, p95 18–21 contra 20–21).
   No es tarea del RL.

#### 4. Sensores

Los tres llegan completos y en el mismo reloj (§4.20, §4.21): GPS e IMU, 100 % de lo
grabado con los mismos valores; frames, 99.9 % a ≤ 1 µs del grabado (0–17 no enviados
por sesión). GPS: 1 Hz sin huecos, velocidad entre fixes contra Doppler 0.22–0.31 m/s
(p50), curso contra giroscopio ±3–8° (p5–p95); no es ground truth (varios metros).
IMU: 53 Hz, intervalo máximo 19 ms. Cámara: 30 fps, 1280×720, JPEG por cable 66–70 ms
de la captura a la llegada; calidad de la VO, punto 2.

#### 5. Evaluación

- **Funciona bien (medido):** sensores y reloj; la fusión con GPS bueno (1–2 m);
  saltos y ráfagas; ruido blanco; el rumbo con el giroscopio, también con lluvia; los
  cortes cortos (10 s); la ventana con mapa en vivo y replay; la caché.
- **Funciona regular:** la latencia en la laptop (p95 167–219 ms > 150; la frecuencia
  cumple); los cortes de más de 20 s (p50 17–161 m a 20–30 s), que dependen de la
  escala de la cámara; la escala (0.16–1.8 veces la Doppler); la compuerta de altos con
  lluvia; el modo solo visual en los altos (§4.21 punto 9); los fixes escasos.
- **Funciona mal:** la cámara sola como trazo (deriva de rumbo, escala); la VO con
  lluvia.
- **Sin medir:** ground truth (no hay); noche; túneles y cortes reales del GPS (solo
  simulados); multitrayecto real (solo sintético); la laptop con el perfil de energía
  "Rendimiento". (Las grabaciones de la de escritorio con el código del 12/10: hechas
  el 13/10, §4.27.)
- **Limitaciones aceptables:** la latencia de la laptop (la de escritorio cumple); la
  cámara sola (es diagnóstico); el multitrayecto (sin otra referencia absoluta no se
  distingue); la VO con lluvia (el rumbo lo da el giroscopio).
- **Para arreglar (no bloquean el RL):** la compuerta de altos con lluvia, y el modo
  solo visual en los altos si se va a reportar como línea base del OE4.

#### 6. ¿Se puede hacer el RL?

**Sí, con un trabajo claro y medible, pero con margen incierto de captar.** El caso con
margen medido es **la confianza en la cámara durante los cortes de GPS** (oráculo: −43 %
de p90 a 30 s); el peso del GPS deja poco tras §4.23, y lo poco lo toma una regla. No
hay que arreglar nada antes. **Riesgo:** con cuatro sesiones (324 cortes) y señales de
correlación ≤ 0.32, lo aprendido puede no generalizar; §3.5.1 vio exactamente eso (6
features peor que no decidir; un umbral sí capturaba). Por eso el RL tiene que medirse
**dejando una sesión afuera** y contra **la regla de velocidad**, no solo contra el EKF
fijo. Más datos ayudan: las tres grabaciones largas de la de escritorio, procesadas allá
y traídas por la caché.

### 4.26 Plan del RL (12/10/2026; propuesta, sin implementar)

**Tarea:** cuánto confiar en la velocidad de la cámara (y, en segundo lugar, en el GPS),
según lo que el filtro observa. Es lo que mide el OE3: "darle más o menos peso a un
sensor según la situación", con estado y recompensa explícitos.

- **Acciones** (discretas, cada 1 s): multiplicador de la varianza de la velocidad de
  la cámara `k_cam ∈ {1, 4, ∞}` (∞ = ignorarla: Doppler o velocidad del estado +
  giroscopio). Fase 2, solo si sobra tiempo: `k_gps ∈ {1, 4}` (6 acciones).
- **Estado** (5 señales, discretizadas en 3 niveles → 243 estados): segundos sin fix
  (0, 1–5, > 5); velocidad del filtro (< 6, 6–12, > 12 m/s); inestabilidad de la razón
  Doppler/cámara en los últimos 10 s; NIS medio de las últimas correcciones de la cámara;
  fracción de frames con la cámara "detenida" en los últimos 5 s. Todas existen hoy en
  `GpsCameraFusion` o en `frames.csv`; se agregan a `OnlineFusion` como un `RlState`
  explícito.
- **Recompensa:** en episodios con el GPS degradado sintéticamente (cortes de 10–60 s,
  ruido, ráfagas; la infraestructura de §4.22), `−‖posición fusionada − GPS limpio‖` en
  cada fix oculto, `−0.1` por cambiar de acción (evita oscilar), y `−` el error con GPS
  bueno si empeora (que no aprenda a dañar el caso normal).
- **Entrenamiento:** Q-learning tabular (243 × 3), fuera de línea, sobre los
  `frames.csv` de la caché (no hace falta procesar video: una sesión completa de
  `run_fusion` tarda ~0.25 s). ε-greedy, unos 20 000 episodios de 60–120 s: ~30–40 min
  de CPU. Tabular a propósito: estado y política se leen en una tabla (explícito para el
  OE3) y no memoriza tanto como un modelo con muchas features (§3.5.1).
- **Evaluación:** dejando una sesión afuera (4 sesiones de campo, más las de la de
  escritorio si llegan por la caché). Mismo protocolo que §4.25: barrido de cortes de
  10/20/30/60 s y el banco de GPS degradado. Comparar contra el EKF fijo, **la regla
  de velocidad** y el oráculo, y reportar qué fracción del margen captura. **Criterio de
  éxito, a declarar antes de entrenar:** capturar ≥ 25 % del margen fijo → oráculo en el
  p90 de los cortes de 20–30 s en las sesiones de prueba, sin empeorar el GPS bueno
  (p95 ± 0.2 m). Si no se logra, se reporta igual: el OE3 pide desarrollar e integrar
  (§11, pregunta 2), y el techo medido es el resultado.
- **Integración:** `rl/policy.py` (estado, acciones, recompensa, la tabla Q en un
  `.npz` versionado); `OnlineFusion` consulta la política cada segundo; `--rl` en
  `run_replay.py` y `run_live.py` y una casilla en la ventana; `metrics.json` guarda la
  distribución de acciones.
- **Costo estimado:** ~7–9 días de trabajo: entorno, estado y recompensa 2 días;
  entrenamiento 1 día; evaluación (reusa el banco y los barridos) 1–2 días; integración
  en el pipeline y la ventana 1–2 días; documentación 1–2 días. Antes de entrenar:
  procesar en la de escritorio las grabaciones largas y commitear la caché (½ día;
  **hecho el 13/10**, §4.27: 6 sesiones para entrenar, 747 cortes de 10–30 s, y
  2025_03_11 para validar), y **congelar el EKF** (el RL aprende sobre el ruido que el
  filtro ve).
- **Ajustes del 13/10 (§4.27):** (1) la evaluación deja **una sesión afuera de seis**;
  2025_03_11 queda aparte, como validación de otro dominio. (2) La regla de velocidad
  ajustada en una sesión empeora otra: la comparación contra ella tiene que ser también
  dejando una sesión afuera, nunca con el umbral ajustado en todas. (3) **El entorno
  necesita el `location.csv` de cada sesión** (la Doppler de cada fix; `frames.csv` no
  la trae, y la IMU no hace falta: `frames.csv` tiene `gyro_yaw`). **Hecho el 13/10:**
  la caché guarda el `location.csv` de cada entrada; las tres de la de escritorio ya lo
  tienen y con solo la caché la fusión da lo mismo que con la grabación (0.0000 m). Las
  cuatro de campo lo agregan en la laptop con
  `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.cache --add-location` (§4.24).


### 4.27 Las grabaciones de la de escritorio, en la caché (13/10/2026)

Procesadas en la de escritorio (RTX 3070, `torch 2.8.0+cu128`) con el código del 12/10
(el mismo hash de código que las cuatro de la laptop), replay determinista completo con
escala por profundidad; en git, `resultados/cache/` (~29 MB con las siete).

| Grabación | Qué es | Opciones | Tiempo | Para el RL |
|---|---|---|---|---|
| `2026_09_30_14_05_19` | Carro, ciudad, 12 min, Doppler p50 2.7 / máx 18.5 m/s, 14 altos | máscara 0.25, intrínsecos del `HELLO`, `--skip-start 6` | 962 s | **Entrenar** |
| `2026_09_30_14_18_36` | Carro, más rápido (p50 6.8, máx 21.7 m/s: lo más rápido que hay), camino mojado | máscara 0.22 (capó), ídem | 843 s | **Entrenar** |
| `2025_03_11` | **Carro** (no caminata: se ve el tablero, Doppler p95 10.8, máx 13.8 m/s), app vieja (MARS Logger), en vertical, fx ≈ 899 | las de la referencia de §13 (máscara 0.14 por defecto) | 283 s | **Solo validar** (otra app, otra orientación y otra cámara) |
| `2026_09_29_13_01_19` | Caminata bajo techo, 90 s | — | — | No sirve: Doppler máx 0.6 m/s, el filtro no arranca |
| `2026_10_09_19_08_06` | El teléfono girado en la mano en el escritorio (§4.20) | — | — | No sirve: Doppler máx 0.2 m/s |

- **La fusión, en las tres:** arranca (23.0, 2.4 y 8.9 s), giroscopio en el 100 % de los
  frames, 0 correcciones de posición rechazadas ni forzadas (695, 767 y 231).
- **Reproduce lo medido antes:** el barrido da en 14:05 y 14:18, a 30 s, 15.7 / 79.1 y
  27.3 / 178.2 m (p50 / p90), los mismos de §4.17.
- **La caché acierta:** el mismo replay de 14:05 otra vez se restauró en 0.6 s, con el
  JSON y el `frames.csv` idénticos byte a byte.
- **La clave se invalida bien** (script temporal, 25 casos sobre una copia de una
  grabación): cambia con un byte distinto en `movie.mp4`, `frame_timestamps.txt`,
  `location.csv` o `gyro_accel.csv`; con cada una de las 9 opciones; con un cambio en
  `pipeline.py`, `depth_scale.py`, `fusion.py` o `sources.py`, y con `CODE_VERSION`. No
  cambia con `movie_metadata.csv` (el replay no lo lee) ni con `run_replay.py`,
  `evaluate.py` o `live_window.py`. **Lo que la clave no ve:** (1) un cambio en el bucle
  de `run_replay.py` que mueva los números, si no se sube `CODE_VERSION` a mano; (2) la
  GPU y las versiones de torch/OpenCV (quedan en `cache.json`: las de la laptop, `cu126`
  en una GTX 1050 Ti; las de aquí, `cu128` en una RTX 3070).

**Episodios para el RL** (cortes de GPS del protocolo de §4.25: uno cada 10 s desde el
arranque, mientras haya fixes después):

| Sesión | Manejo con el filtro | 10 / 20 / 30 / 60 s | 10–30 s |
|---|---|---|---|
| Campo, 4 sesiones (§4.25) | 19.9 min | 113 / 109 / 105 / 93 (reconstruidos) | **324** (327 reconstruidos desde la caché) |
| 14:05 | 11.6 min | 68 / 67 / 66 / 63 | 201 |
| 14:18 | 12.8 min | 75 / 74 / 73 / 70 | 222 |
| **Total para entrenar (6 sesiones)** | **44 min** | 256 / 250 / 244 / 226 | **747** (2.3 veces) |
| 2025_03_11 (validación) | 4.7 min | 22 / 21 / 20 / 17 | 63 |

**El margen del RL en las sesiones nuevas** (mismo protocolo que §4.25; script temporal:
en cada corte, el EKF tal cual contra el mismo filtro sin la velocidad de la cámara
durante el corte; oráculo = el mejor de los dos; p50 / p90 en m):

| | 10 s | 20 s | 30 s | 60 s | Sin cámara es mejor en |
|---|---|---|---|---|---|
| 14:05 EKF → oráculo | 4.9 / 23.0 → 4.6 / 17.2 | 9.8 / 49.1 → 8.3 / 32.1 | 15.7 / 79.1 → 14.4 / 53.6 | 25.1 / 186 → 24.4 / 108 | 24–37 % |
| 14:18 EKF → oráculo | 10.6 / 42.7 → 6.7 / 22.6 | 16.9 / 107 → 15.0 / 38.7 | 27.3 / 178 → 16.7 / 77.0 | 83.2 / 447 → 49.2 / 218 | 34–43 % |
| 2025_03_11 EKF → oráculo | 12.4 / 35.1 → 11.7 / 33.9 | 23.9 / 50.0 → 22.2 / 48.2 | 32.1 / 84.9 → 31.1 / 75.9 | 61.8 / 147 → 54.2 / 108 | 29–45 % |

- **El margen se confirma:** a 30 s, p90 −32 % en 14:05 y −57 % en 14:18 (campo, −43 %).
  Es mayor en la sesión rápida, como en el campo (150517). En 2025_03_11 es chico a
  10–30 s (−3 a −11 %) y aparece a 60 s (−27 %).
- **La regla de velocidad no se traslada de una sesión a otra** (hallazgo central de
  esta sección). La regla: durante un corte, ignorar la velocidad de la cámara si el
  filtro va a más de un umbral; el umbral se elige para el mejor p90 en una sesión y
  se prueba en la otra (script temporal):

  | Corte | Ajustada en 14:05 → probada en 14:18 | Ajustada en 14:18 → probada en 14:05 |
  |---|---|---|
  | 10 s | umbral 10.0 m/s; p90 42.7 → 25.4 (oráculo 22.6): **86 %** del margen | umbral 13.2; p90 23.0 → 24.8 (oráculo 17.2): **−32 %** |
  | 20 s | 13.3; 106.5 → 73.0 (38.7): **49 %** | 8.8; 49.1 → 60.7 (32.1): **−69 %** |
  | 30 s | 13.3; 178.2 → 126.2 (77.0): **51 %** | 7.3; 79.1 → 108.5 (53.6): **−115 %** |
  | 60 s | 13.3; 447.3 → 341.0 (217.9): **46 %** | 7.3; 185.8 → 237.3 (107.5): **−66 %** |

  Un sentido captura la mitad del margen o más; el otro **empeora al EKF fijo**: en
  14:05 la cámara sirve a velocidades donde en 14:18 ya no (14:18 es más rápida y con
  el camino mojado). La señal existe en las dos (correlación de la velocidad con
  "conviene sin cámara" +0.38 a +0.54 en 14:18, +0.17 a +0.27 en 14:05; campo, +0.32),
  pero su punto de corte depende de la sesión. Es el patrón de §3.5.1 y §4.25 medido
  directamente entre dos sesiones. **Qué implica:** (1) una regla fija de una señal no
  es una línea base confiable fuera de la sesión donde se ajustó, lo que refuerza el
  argumento del RL: decidir con varias señales a la vez (velocidad, inestabilidad de
  la razón Doppler/cámara, NIS de la cámara, segundos sin fix); (2) el mismo riesgo vale
  para el RL: tiene que medirse dejando una sesión afuera, y contra la regla de
  velocidad evaluada igual (ajustada sin la sesión de prueba), nunca contra una regla
  ajustada con todas.
- **Lo que suman al plan:** más del doble de cortes y de manejo, y las velocidades
  altas (14:18 llega a 21.7 m/s; el campo, a 17.9) para el nivel "> 12 m/s" del estado.


### 4.28 El entorno del RL (13/10/2026)

`realtime/rl/env.py` (nuevo). Corre sobre la caché, sin video; `python -m
LMS.LMS_RL_ORB_GPS.realtime.rl.env` lo revisa (identidad con el EKF fijo y los cortes con
`k` constante) y sale con error si la identidad se rompe.

- **El filtro:** `PolicyFusion`, subclase de `GpsCameraFusion`; `fusion.py` no se tocó
  (la caché sigue valiendo para cualquier política). Solo cambia el peso de la
  velocidad de la cámara: la varianza (en marcha o detenida) por `k ∈ {1, 4, ∞}`, con
  ∞ = no usarla. **Con `k = 1` es el EKF fijo: 0 m de diferencia** en toda la sesión, en
  las tres; con un error a propósito (la sonda del NIS sobre el filtro real en vez de
  una copia) la diferencia sube a 1.47 m: la prueba detecta que la subclase toque el
  filtro.
- **Cuándo decide:** una vez por segundo, en cada velocidad de la cámara, y **solo con
  1.5 s o más sin fix**; con el GPS presente decide el filtro fijo. Medido: con el GPS
  bueno la velocidad de la cámara no aporta (ignorarla en toda la sesión da la
  predicción contra cada fix p50/p95 1.25/3.57 contra 1.30/3.88 m en 14:05, 1.28/3.29
  contra 1.33/3.33 en 14:18, 2.42/5.55 contra 3.24/6.68 en 2025_03_11). Así "no
  empeorar el GPS bueno" se cumple por construcción y la recompensa no necesita ese
  término; el estado del filtro al empezar cada corte es el del EKF fijo.
- **Episodio:** una copia del filtro fijo cada 1 s de la sesión (`Snapshots`); un corte
  de 10–60 s desde una de ellas, hasta el fix con que vuelve el GPS. La grilla de
  evaluación (uno cada 10 s desde el arranque) es la de `evaluate.py --sweep`.
  ~30 episodios por segundo.
- **Estado (`RlState`, 3⁵ = 243):** segundos sin fix (≤ 5, ≤ 15, más); velocidad del
  filtro (≤ 6, ≤ 12, más m/s); dispersión de Doppler/cámara en los últimos 10 fixes
  (coeficiente de variación; ≤ 0.18, ≤ 0.25, más); NIS medio de las últimas 5
  velocidades de la cámara, calculado sobre una copia con el peso del filtro fijo, así
  que sigue llegando mientras se la ignora (≤ 0.2, ≤ 0.75, más); fracción de frames
  "detenidos" por la compuerta en los últimos 5 s (0, ≤ 0.5, más). La dispersión y el
  NIS van en sus terciles sobre los cortes de 14:05 y 14:18 (con la χ² de 1 grado el
  74 % del NIS caía en el primer nivel); hay que revisarlos cuando entren las de campo.
  En la grilla: 19 109 decisiones, 137 estados visitados, 119 con ≥ 20 decisiones.
- **Recompensa:** −(distancia a cada fix oculto y al de la vuelta), en m, −0.1 por
  cambio de acción (§4.26). La suma por decisión coincide con la del episodio (5·10⁻¹³).
  Tiene colas grandes: suma por episodio p50 −56 (10 s) a −1630 m (60 s), p10 −2783.
  **Opción:** la misma contra el EKF fijo del mismo corte (cuánto más cerca que él, fix
  por fix): elige la misma política (lo que se resta no depende de las acciones) y le
  quita la dificultad de cada tramo; con una política al azar, en cortes de 30 s, el
  rango p10–p90 baja de −2371…−148 a −206…+146 (14:18).

**Con `k` constante en todo el corte** (desde los 1.5 s sin fix; p50 / p90 al volver el GPS, m):

| | 10 s | 20 s | 30 s | 60 s |
|---|---|---|---|---|
| 14:05 EKF fijo / `k = 4` / sin cámara / oráculo de los 3 | 4.9/23.0 · 4.5/21.6 · 9.6/30.8 · **3.9/16.9** | 9.8/49.1 · 11.3/43.4 · 26.3/78.9 · **9.2/30.9** | 15.7/79.1 · 14.1/79.2 · 46.7/115 · **11.5/49.0** | 25.1/186 · 30.4/173 · 88.0/279 · **20.3/94.2** |
| 14:18 | 10.6/42.7 · 10.3/33.1 · 16.0/37.9 · **6.2/19.5** | 16.9/107 · 19.6/114 · 31.5/76.7 · **14.3/38.7** | 27.3/178 · 28.1/181 · 48.6/128 · **15.7/79.8** | 83.2/447 · 80.6/509 · 97.6/334 · **45.9/205** |
| 2025_03_11 (validación) | 12.4/35.1 · 11.5/36.0 · 18.2/52.0 · **11.1/34.8** | 23.9/50.0 · 21.1/75.8 · 40.4/118 · **20.2/48.6** | 32.1/84.9 · 30.6/116 · 50.8/173 · **29.1/84.9** | 61.8/147 · 61.0/157 · 96.6/284 · **53.6/110** |

- El EKF fijo coincide con §4.27; "sin cámara" difiere un poco de allá porque ahora el
  primer 1.5 s del corte lo decide el filtro fijo.
- **`k = 4` solo, casi igual al EKF fijo; en el oráculo suma algo:** es el mejor de los
  tres en el 30–45 % de los cortes de 30 s. Con la misma restricción, el oráculo de 2
  acciones (1 e ∞) da p50 / p90 14.0 / 51.1 (14:05), 19.0 / 80.8 (14:18) y 31.1 / 84.9
  (2025_03_11); con 3, 11.5 / 49.0, 15.7 / 79.8 y 29.1 / 84.9: baja la mediana 2–3 m y
  el p90 0–2 m.
- **Las señales al empezar el corte** (correlación con "conviene sin la cámara"):
  velocidad +0.20 / +0.48 (14:05 / 14:18), dispersión +0.13 / +0.31, NIS +0.13 / +0.31,
  detenido −0.09 / −0.27. **En 2025_03_11 ninguna** (−0.15 a +0.18 sin patrón): otro
  dominio; puede que el RL no gane ahí, y se reporta igual.

**Cortes reales (preguntas del estudiante, 13/10):** la app usa el **proveedor
fusionado** de Android (`FusedLocationProviderClient`, `LocationProvider.java`), no el
GPS solo. En un túnel puede seguir mandando posiciones estimadas (red, inercial) en vez
de cortarse, y saltar a la salida; sin la precisión de cada fix (`getAccuracy`, deseable
de §10) el filtro no lo distingue. Si los fixes no se cortan, la política (que decide
con ≥ 1.5 s sin fix) no se activa y lo que actúa es el rechazo de saltos del EKF fijo
(§4.18, §4.23). **No medido.** Propuesta: grabar el túnel Zurquí (Ruta 32) en los dos
sentidos y el centro de San José, como caso de prueba real (no para entrenar): cómo
falla el GPS del teléfono, la cámara con la luz del túnel, y el error al salir contra el
primer fix bueno, que es la misma métrica de los cortes simulados.


### 4.29 ¿El RL también para el peso del GPS? No (13/10/2026)

Pregunta del estudiante: el ruido del GPS sí está en las grabaciones y los cortes no;
¿conviene que el RL decida también cuánto confiar en el GPS (su varianza)? Banco temporal
sobre la caché (14:05, 14:18, 2025_03_11; 3 semillas): ruido blanco de σ 0/5/10/20 m en
cada eje de los fixes, la Doppler intacta; error de la posición fusionada contra el fix
limpio en cada fix desde el arranque. La varianza del GPS se multiplica por `k` desde
una subclase (atributo de la instancia), sin tocar `fusion.py`. El oráculo elige cada
10 s el mejor `k ∈ {1, 4, 16}` **mirando el GPS limpio**: es una cota optimista.

| p50 / p95 (m) | GPS solo | EKF fijo | `k = 4` fijo | `k = 16` fijo | Regla del NIS (§4.25) | Oráculo cada 10 s |
|---|---|---|---|---|---|---|
| GPS limpio, 14:05 · 14:18 | 0 | 1.0/3.0 · 1.0/2.7 | 1.5/4.0 · 1.4/3.6 | 2.4/5.4 · 1.9/4.8 | 1.0/3.0 · 1.0/2.7 | 1.0/3.0 · 1.0/2.7 (elige `k = 1` el 96–100 %) |
| σ 5 m | 5.9/12.3 | 2.0/4.9 · 2.1/4.9 | 2.2/5.7 · 2.5/5.7 | 3.4/9.4 · 3.8/10.0 | 2.0/5.4 · 2.3/5.3 | 1.7/4.6 · 1.9/4.6 |
| σ 10 m | 11.8/24.6 | 4.4/11.0 · 4.7/11.1 | **3.0/7.7 · 3.4/8.0** | 3.6/8.9 · 4.1/11.3 | 3.2/8.8 · 3.7/8.5 | 2.3/7.0 · 2.7/7.0 |
| σ 20 m | 23.6/49.3 | 14.1/38.6 · 15.2/39.0 | 8.3/19.6 · 7.4/19.9 | **4.8/26.9 · 5.5/15.0** | **1146/1542** · 5.7/13.2 | 3.8/12.7 · 4.3/10.9 |

2025_03_11, igual de forma: limpio 2.8/5.9 (oráculo el mismo), σ 10 m 7.8/14.7 →
oráculo 5.9/11.0, σ 20 m 17.5/36.5 → 8.2/25.5 (la regla del NIS: 224/347).

**Por qué no:**
1. **Con el GPS de las grabaciones no hay margen.** Con el GPS limpio, cualquier `k > 1`
   empeora y el oráculo elige el filtro fijo el 96–100 % del tiempo. El ruido natural es
   chico: el NIS de posición es 0.17–0.49 (§4.25), menos que los 4 m que supone el
   filtro, y la predicción contra cada fix es 1.3–2.0 m (p50). Con σ 5 m tampoco: el EKF
   fijo ya filtra a 2.0/4.9 contra 5.9/12.3 del GPS solo, y el oráculo baja 0.3 m.
2. **El ruido natural no se puede evaluar.** La única referencia es el mismo GPS: una
   política que confíe más en él (`k < 1`) se "acerca" a la referencia por definición.
   Para medir hay que sumar ruido sintético sobre fixes limpios, igual que los cortes:
   el argumento de "el ruido sí ocurre de verdad" no favorece al ruido sobre los cortes.
3. **Donde hay margen (σ ≥ 10 m), lo toma un peso fijo según el nivel de ruido.** A σ 10
   m, `k = 4` en toda la sesión da 3.0–3.4/7.7–8.0 contra el oráculo 2.3–2.7/7.0; a σ 20
   m, `k = 16` llega a 4.8–5.5/15–27 contra 3.8–4.3/11–13. La decisión es "cuánto ruido
   tiene este GPS", que cambia despacio: un problema de **estimación** (un filtro de
   Kalman adaptativo, con la varianza estimada de las innovaciones), no de decisiones
   secuenciales. Para el RL quedaría poco por encima de esa regla.
4. **Una regla adaptativa ingenua puede divergir:** con σ 20 m la regla del NIS se fue a
   1146 m (14:05) y 224 m (2025_03_11): varianza alta → innovaciones más grandes →
   varianza más alta, hasta soltar el GPS. En 14:18 funcionó (5.7/13.2). Si algún día se
   adapta el peso del GPS, tiene que ir acotado (acciones discretas, como las del RL).
5. **El ruido real de ciudad es multitrayecto** (error lento y correlado): ningún peso lo
   corrige (§4.22, §4.25).

**Si se quisiera de todos modos** (no recomendado; solo si sobra tiempo): acciones
`k_gps ∈ {1, 4, 16}` junto a las de la cámara (9 acciones); al estado, el NIS medio de la
posición y la fracción de fixes rechazados recientes (cada uno 3 niveles: 243 × 9 = 2187
pares, demasiados para 6 sesiones; habría que sacar señales); episodios con el nivel de
ruido sorteado sobre la caché y la recompensa contra el fix limpio. El margen esperado,
por el oráculo: σ 10 m, p95 de ~11 a ~7 m; σ 20 m, de ~39 a ~11–13 m; con GPS limpio o σ
5 m, nada.

**Decisión: el RL queda en la confianza en la cámara durante los cortes (§4.26, §4.28).**

---

## 5. BUGS IDENTIFICADOS EN `main.py` (SIN CORREGIR)

1. **Calibración equivocada para móvil.** `main.py:29` usa por defecto los valores
   de KITTI (`fx=718.856, cx=607.19, cy=185.22`, imagen 1242×375) y
   `gui_processor.py:93` instancia sin parámetros. El video del teléfono es
   720×1280 **vertical** con `fx≈899`. **Medido: corregirlo baja el error de la VO
   de 27.18 m a 11.01 m (−59.5 %).**
2. **`Dropout(0.1)` activo en inferencia.** El único `.eval()` está en
   `scripts/training/validate_rl_behavior.py:51`. **Todos los resultados con RL son
   no deterministas.**
3. **`visual_conf` es una constante** (§3.4).
4. **El tablero del carro ocupa la parte baja del frame** en los videos móviles.
   Genera correspondencias que dicen "no te moviste". El pipeline nuevo no busca
   puntos ahí (máscara del 14 % inferior). Recortar la imagen, como se hacía al
   principio, le cambiaba la escala al modelo de profundidad (§4.12).
5. `README_ORGANIZACION.md` desactualizado.
6. Los imports de `main.py` hacen que `python -m LMS.LMS_RL_ORB_GPS.main` **falle**.
7. **`optimize_pose_graph()` no optimiza nada** (`LMS_ORB_with_PG/main.py:99`).
   Su residuo es `(p[i] − p[i−1]) − movimiento_medido[i−1]`, solo entre poses
   consecutivas. El mínimo global de esos mínimos cuadrados **reproduce
   exactamente la cadena de VO de entrada** (residuo cero alcanzable). Sin loop
   closures ni restricciones absolutas, un pose graph no tiene nada que
   corregir. El nombre `PoseGraphSLAM` es aspiracional.
8. **No existe loop closure, relocalización ni bundle adjustment** en ningún
   archivo del repositorio. Verificado por búsqueda. Relevante para §11.1.
9. **`recoverPose` con su umbral de distancia por defecto** (`main.py:323`,
   `LMS_ORB_with_PG/main.py:69`, los otros `main.py` y los scripts de análisis).
   Con poco desplazamiento entre frames, la elección entre las cuatro soluciones
   empata y la mitad de las veces devuelve un giro de 180°. Es la basura con el
   vehículo detenido de §4.6. Corregido en `realtime/` (§4.13).

---

## 6. ERRORES COMETIDOS EN EL ANÁLISIS (para calibrar confianza)

Se cometieron **doce errores en los análisis** (los cinco primeros y el octavo,
de programación), y cada uno cambió o invirtió las conclusiones:

1. **Rotación inicial identidad.** El marco de la cámara de KITTI está girado
   ~124° respecto del mapa. Corregido con `estimate_initial_rotation()`.
2. **GPS en cada frame.** Régimen irreal donde no hay nada que adaptar.
3. **Desajuste de distribución.** Features calculadas sobre la trayectoria del
   oráculo pero aplicadas sobre la de la política.
4. **Base de triangulación de 1 frame** en el sondeo de profundidad, que produjo
   un falso negativo (§3.2).
5. **Desalineo de índices al comparar contra el GPS** (`run_replay.py`,
   8/9/2026). La serie de GPS se filtraba con `[g for g in gps if g is not None]`
   sin conservar la posición: los 19 frames iniciales sin fix corrían toda la
   serie 19 lugares (~0.6 s, ~4 m a la velocidad de este recorrido). Inflaba el
   RMSE de 15.08 a 16.72 m. Corregido emparejando por índice.
6. **Ajuste del giroscopio frame a frame** (29/9/2026). La orientación IMU →
   cámara ajustada sobre frames sueltos no salió una permutación de ejes y dio un
   desfase falso de +32 ms, porque frame a frame el ruido de la VO es del tamaño
   de la rotación real. Sobre ventanas de 0.5 s el ajuste queda limpio (§4.10).
7. **Leer la calidad de la rotación en la razón de inliers** (29/9/2026). Se dijo
   que la mitad de los frames de la caminata del 29/9 "no tenía geometría"; el giroscopio
   mostró que esos frames tenían el error más bajo.
8. **Encadenamiento de poses al revés** (heredado del sistema original; descubierto
   el 2/10/2026, §4.11). Afectó todos los mapas y las trayectorias de los scripts
   de análisis; la alineación 3D lo escondía. Se encontró al evaluar en el plano
   horizontal contra el giroscopio, donde ya no se puede esconder.
9. **Resumir el rumbo sumando con signo** (2/10/2026). "−2° en marcha" eran saltos
   que se cancelaban. Para atribuir error hay que mirar la curva o sumar magnitudes.
10. **Atribuir los saltos de rumbo a los altos** (2/10/2026). §4.12 decía "sobre
   todo en altos y a baja velocidad"; en 14:05, 19 de los 31 pasos de más de 5°
   fueron a más de 3 m/s. La causa era un empate dentro de `recoverPose` (§4.13),
   que depende del desplazamiento entre frames comparado con la profundidad de la
   escena, no de la velocidad sola. Se encontró listando cada paso malo con su
   número de inliers en vez de mirar promedios.
11. **Generalizar la máscara desde un solo tramo** (2/10/2026). La prueba de concepto
   (0.62 contra 0.92) se hizo en 70 s de 14:05, y se escribió que en 14:18 "el
   recorte no sesgaba la escala" sin medirlo con la imagen completa. Medido: la
   máscara mejora 14:05 (−30 % a +10 %) y empeora 14:18 (+2 % a +59 %) y la
   referencia vieja (+8 % a +34 %). Es la regla de §13: no declarar éxito con una
   sola secuencia.
12. **Recomendar la red de seguridad de 20° con datos viejos** (3/10/2026). Se dijo
   que los fallos que atrapa eran "de 97° o más": era la lista de antes de la
   corrección del empate. Con la votación nueva, los tres que atrapaba en 14:05
   son de 11–15° (§4.13). El estudiante aprobó 20° con esa premisa; se detectó al
   escanear los pasos después de aplicar el cambio.

También hubo **mediciones en frío tomadas como típicas**: la primera inferencia del
modelo (4.5 s) y su carga (30 s) resultaron ser arranques en frío; lo normal es
0.3 s y 4.5 s.

**Un pronóstico que resultó equivocado, para calibrar:** se supuso que comparar
frame a frame contra un GPS de 1 Hz inflaba mucho la métrica (entre fix y fix la
posición del GPS se congela mientras el vehículo avanza — es el diente de sierra
visible en la curva de desviación). Se midió comparando **solo en el instante en
que entra cada fix**: el efecto es de ~1 m sobre 15, no de varios metros. El
diente de sierra es visible pero casi no mueve el número. Esa métrica quedó
implementada como línea adicional en la salida.

También hubo una **estimación de esfuerzo equivocada** sobre el plano de tierra
("2-3 semanas" cuando es trabajo de investigación bastante mayor), y una
**conclusión prematura** al declarar éxito de la profundidad con una sola
secuencia.

**`scripts/rl/sanity_checks.py` existe para que estos errores no se repitan.
Correrlo antes de confiar en cualquier número nuevo.**

---

## 7. INVENTARIO DE SCRIPTS

> **Reorganizado el 8/9/2026.** Las sondas de escala no eran RL: son la
> investigación que produjo `realtime/depth_scale.py`. Se separaron en
> `scripts/scale/`. `oracle_analysis.py` se eliminó (obsoleto y con el bug de
> rotación inicial); sigue en el historial de git si hiciera falta.

### `scripts/rl/` — RL y su cadena de verificación

| Script | Estado | Para qué |
|---|---|---|
| `sanity_checks.py` | **Importante** | 6 invariantes + auditoría de fugas |
| `extract_features.py` | Sirve | Cachea VO+GPS+GT de KITTI |
| `realistic_rates.py` | Sirve | Oráculo con GPS a 1 Hz. Exporta `estimate_initial_rotation` |
| `imitate_oracle.py` | Parcial | Experimento fallido, pero `sanity_checks` importa su `run_episode` |

Los tres últimos existen porque `sanity_checks.py` depende de ellos. Borrar
cualquiera deja sin correr los 6 invariantes.

### `scripts/scale/` — investigación de escala métrica (no es RL)

| Script | Estado | Para qué |
|---|---|---|
| `depth_scale_probe.py` | Sirve | Precisión de escala por profundidad (§3.3). Acepta varias secuencias y guarda features observables |
| `scale_decision.py` | **Importante** | La puerta del OE3 (§3.5.1): margen y aprendibilidad |
| `depth_trajectory_test.py` | Sirve | Validó en KITTI con ground truth (§3.3) |
| `ground_plane_probe.py` | Histórico | Método descartado; evidencia de §12.6 |

`depth_trajectory_test.py` (aquí) y `realistic_rates.py` y `sanity_checks.py` (T4)
en `scripts/rl/` encadenan las poses al revés (§4.11). No se corrigieron.

### `realtime/` — el sistema

| Módulo | Estado |
|---|---|
| `sources.py`, `pipeline.py`, `run_replay.py` | Probados. Desde el 2/10/2026 el front-end visual usa máscara en vez de recorte, vota con todos los puntos y descarta giros de más de 10° (§4.13) |
| `run_live.py` | Probado con el simulador y con el teléfono (§4.9); verificado de nuevo el 5/10/2026 con el código actual |
| `phone_simulator.py` | Nuevo (5/10/2026): el teléfono simulado, para probar el modo en vivo y la interfaz sin el celular |
| `session.py`, `live_window.py` | Nuevos (5/10/2026): la sesión reutilizable y la ventana; `run_live.py` quedó como su consola. Probados en Xvfb con el simulador y con grabaciones, contando ventanas |
| `evaluate.py` | Nuevo (2/10/2026): evalúa un `frames.csv` contra el GPS y el giroscopio de la grabación (§4.12); desde el 4/10/2026 también la fusión (§4.14) |
| `fusion.py` | Nuevo (4/10/2026): el EKF; probado sin conexión sobre los dos fragmentos (§4.14) |
| `depth_scale.py` | Validado e **integrado** (`DepthScaleWorker` en `pipeline.py`). Desde el 4/10/2026 reporta 0 m/s con el carro detenido (§4.15) |

---

## 8. DATOS DISPONIBLES

### KITTI (`kitti_data/2011_09_26/`)
8 secuencias, ~3374 frames. `drive_0048` inservible (22 frames).

| Secuencia | Frames | Distancia | m/frame |
|---|---|---|---|
| 0001 | 108 | 107 m | 1.05 |
| 0005 | 154 | 69 m | 0.42 |
| 0009 | 447 | 333 m | 0.92 |
| 0013 | 144 | 173 m | 1.18 |
| 0018 | 270 | 51 m | **0.00** (detenido) |
| 0071 | 1059 | 240 m | 0.23 |
| 0042 | 1170 | 2591 m | 2.45 |

OXTS a **10 Hz** en las carpetas `_sync`. `vf` y las orientaciones son la
**salida del filtro GNSS/INS**, no IMU cruda — usarlas reintroduce dependencia del
GPS. El RT3003 es de **grado navegación**, no comparable con un MEMS de teléfono.
Los archivos de calibración **no están descargados**.

El 2/10/2026 se decidió borrar las carpetas que el pipeline no usa
(`velodyne_points`, `image_00`, `image_01`, `image_03`, ~10 GB) para liberar
disco. Quedan `image_02` y `oxts`. Se pueden volver a descargar.

### Móvil (`mobile_data/2025_03_11/`)
Grabado con una versión modificada de **MARS Logger** (Android). **Es un recorrido en
carro, no una caminata** (se ve el tablero; Doppler p50 2.9, p95 10.8, máximo 13.8 m/s;
1.8 km según la cámara). La caminata es `2026_09_29_13_01_19` (§4.10). Para el RL va
solo como validación: otra app, el teléfono en vertical, otros intrínsecos (§4.27).

| Propiedad | Valor |
|---|---|
| Video | 720×1280 vertical, 29.97 fps, 8673 frames |
| fx | ≈899 px, desv. 1.51 (muy estable) |
| Obturador rodante | 20.8 ms |
| GPS | **1.00 Hz exacto** (240 muestras / 239 s) |
| IMU | 49.9 Hz |

No hay calibración extrínseca cámara–IMU. El tablero aparece en el cuadro.

### Grabaciones de la app nueva (Galaxy S23)

Formato en `ESTADO_APP_ANDROID.md` §5. Lo que importa para el pipeline:
- La primera columna de cada archivo está en el reloj de arranque; es la que hay
  que usar para alinear (la columna Unix de la cámara es la llegada a la app).
- 1280x720 horizontal. Los intrínsecos son los del `HELLO` (fx 867.81, fy 868.55,
  cx 630.75, cy 367.79) y **la grabación no guarda cx ni cy**.
- Los primeros ~6 frames del video los escribe el codificador antes de ponerse al
  día (en la del 29/9 los tres primeros son idénticos): `--skip-start 6`.

| Grabación | Dónde | Qué es |
|---|---|---|
| `2026_09_29_13_01_19` | `mobile_data/` (89 MB) | Caminata bajo techo, 90 s, vuelve al punto de partida (§4.10) |
| `2026_09_30_14_05_19` | `mobile_data/` (695 MB) | Carro, 12.0 min, 3.3 km, **14 altos** (109 s), v p50 2.7 m/s, máx 67 km/h. Máscara 0.25 |
| `2026_09_30_14_18_36` | `mobile_data/` (732 MB) | Carro, 12.8 min, 6.6 km, 6 altos (71 s), v p50 6.8 m/s, máx 78 km/h. Máscara 0.22 |

Las del carro se trajeron del teléfono el 2/10/2026 (quedaron 16 GB libres en el
disco). Son de día (exposición ~1.3 ms, ISO 50), traen giroscopio, no son
recorridos cerrados (inicio y fin a 2.3 y 5.5 km) y quedan en la zona UTM 16. En
la imagen se ven el **capó, que refleja la escena** (desde el 25 % y el 22 %
inferior) y el tablero (desde el 15 %). En la del 14:18 el camino está mojado.
**Su GPS va a 1.00 Hz**, no a los 1.9 Hz que midió el task de la app caminando.
La grabación `2026_09_30_14_04_17` (6 s) es un arranque fallido.

Para reproducirlas (la máscara depende de si se ven el capó o el tablero):

```
venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay mobile_data/<carpeta> \
    --fx 867.81 --fy 868.55 --cx 630.75 --cy 367.79 --mask-bottom 0 --skip-start 6
```

---

## 9. HARDWARE

| | |
|---|---|
| CPU | AMD Ryzen 5 5600X, 12 hilos |
| GPU | **NVIDIA RTX 3070, 8 GB — driver instalado y funcionando** |
| torch | 2.8.0+cu128, CUDA disponible |
| Modelo de profundidad | 66–150 ms por frame en GPU (optimizable con fp16) |

**Laptop de campo (confirmado el 11/10/2026):** Intel i5-8300H (4 núcleos, 8 hilos),
GTX 1050 Ti 4 GB (sm_61, driver 580), Linux Mint 22, `torch 2.8.0+cu126` (la de PyPI,
`+cu128`, no trae código para sm_61; `requirements.txt` ya fija la de CUDA 12.6, §15).
Modelo de profundidad 291 ms p50 en su GPU. **Por frame ~50 ms (el doble que el
escritorio): ~19 Hz efectivos, 36 % de descarte y captura a resultado p95 167–189 ms
por cable** (§4.20).

---

## 10. SIGUIENTES PASOS (plan vigente al 7/10/2026)

### Dónde quedamos (9/10/2026)

**13/10/2026, el entorno del RL (§4.28):** `realtime/rl/env.py`, sobre la caché y sin
tocar `fusion.py`; con `k = 1` es el EKF fijo a 0 m. La política decide solo con el GPS
ausente (con el GPS bueno la cámara no aporta: medido). Estado de 243 con niveles
parejos, recompensa verificada; el oráculo de 3 acciones a 30 s, p90 79 → 49 m (14:05)
y 178 → 80 m (14:18). **Espera aprobación del estudiante** (la restricción a los cortes,
la recompensa contra el EKF fijo y los niveles) antes del entrenamiento. **El peso del
GPS no entra en el RL** (§4.29): con el GPS de las grabaciones no hay margen ni forma de
medirlo, y donde hay (ruido sintético σ ≥ 10 m) lo toma un peso fijo por nivel de ruido.

**13/10/2026 (§4.27):** procesadas y en la caché las grabaciones de la de escritorio:
14:05 y 14:18 para entrenar (con las cuatro de campo, 6 sesiones, 44 min de manejo,
747 cortes de 10–30 s), 2025_03_11 solo para validar (es carro con la app vieja, no
caminata); la caminata del 29/9 y la prueba con el teléfono quieto en el escritorio del
9/10 no sirven (el filtro no arranca). La clave de la caché se invalida bien (25 casos). El margen del RL se confirma
en las sesiones nuevas (30 s: p90 −32 % y −57 %); la regla de velocidad no se traslada
entre sesiones: ajustada en una, empeora la otra (§4.27, §12 punto 21). La caché guarda
el `location.csv` de cada entrada (con eso la fusión corre sin la grabación); falta
agregarlo a las cuatro de campo en la laptop (`git pull` y `python -m
LMS.LMS_RL_ORB_GPS.realtime.cache --add-location`). **Siguiente:** el RL, empezando por
el entorno, el estado y la recompensa, con el EKF congelado (§4.26).

**12/10/2026 (§4.23–4.26):** el curso del GPS condicionado a la Doppler (y sin forzar
con el giroscopio) y las ráfagas sin forzar: la fusión ahora filtra el GPS ruidoso (σ
10 m: p95 12–16 m contra 24) y no sigue las ráfagas (4 m contra 52), sin cambiar nada
con GPS bueno. Mapa también en replay; caché de replays deterministas (`cache.py`, en
git) con las cuatro sesiones de campo. Diagnóstico general medido (§4.25) y plan del RL
(§4.26): la tarea es la confianza en la cámara durante los cortes de GPS (oráculo −43 %
de p90 a 30 s; ninguna regla simple lo captura); ~7–9 días. **Siguiente:** procesar en
la de escritorio las grabaciones largas y commitear su caché; congelar el EKF; el RL.

**11/10/2026, prueba de campo (§4.21):** cuatro sesiones en ciudad (1–3.3 km), las
grabaciones en `mobile_data/2026_10_11_*`. Cámara, GPS e IMU llegaron completos (lo
recibido es idéntico a lo grabado); el EKF arrancó en todas, giró con el giroscopio en el
~100 % de los frames y aceptó todas las correcciones de posición. La cámara sola sigue
sin servir como trazo (deriva de rumbo y escala errática) y con lluvia en el parabrisas
no ve los giros. El giro de 180° de 151614 es real (vuelta a la cuadra, confirmado
con el video). El gráfico de cada sesión ya trae flechas de sentido, tiempo y el
trazo cámara + giroscopio (§4.21, punto 8); la ventana tiene el mapa con la fusión y el
GPS (paso 3, punto 10). Con el GPS degradado (§4.22) la fusión aguanta saltos sueltos y
pérdidas moderadas, pero con ruido queda peor que el GPS solo porque el peso del GPS es
fijo y el curso entre fixes confía en cada par; con el peso ajustado filtra el ruido a
la mitad (motivación medida para el RL). Pendiente: el modo solo visual se pasa en
los altos cuando la compuerta rechaza los de la cámara (punto 9); la escala de la cámara (cámara/Doppler 0.6–1.8 en campo) y
la varianza del GPS en el filtro (NIS 0.17–0.49, esperado 2).

**11/10/2026, la laptop de campo (§4.20, §14 "Verificación en la laptop"):** lista,
con torch `+cu126`. Por cable funciona, pero la latencia p95 no cumple en ella (167–189
ms; el cómputo por frame es el doble que en la de escritorio). Decisión: la prueba de
campo va con la configuración actual. Antes o durante la prueba, medir el trazo y el
EKF con el descarte real (~37 %), que nunca se midió: `--strict --trajectory` sobre
`2025_03_11` en la laptop, o la grabación de la prueba.

0. **9/10/2026, sin commitear:** las líneas de la IMU y la fusión en la ventana y la
   consola, y la consulta de hora en el pipeline (§4.20): `fusion.py`, `session.py`,
   `live_window.py`, `run_live.py`, `run_replay.py`, `sources.py`, `pipeline.py`,
   `phone_simulator.py`. El paso 2 quedó commiteado (`516f2b1`).

1. **Commiteado hasta `579eb5a`** (9/10/2026): el paso 2, el EKF dentro del pipeline
   (§4.19), en `516f2b1`; B y C en `28a5e2e`; el paso 1 en `6df67e4`.
2. **Decidido el 7/10/2026, con el asesor: el rumbo sale del giroscopio** (§4.16). El
   giro cada 10 frames queda descartado como fuente principal (al final, solo si sobra
   tiempo). En la misma revisión del EKF, **la velocidad pasa a estado** y **el sesgo
   del giroscopio se estima con el curso del GPS**, los dos medidos con y sin. El
   alcance "sin IMU" deja de regir y el margen del RL se achica: las dos cosas, aceptadas
   por el estudiante. La app tiene que transmitir la IMU: pedido firme en "Pedidos
   pendientes para el task de la app", que el estudiante pasará a ese task.
3. **Plan, en este orden** (estimados en "Orden acordado"; antes de commitear cada paso,
   el estudiante revisa sus números):
   0. Consola de `run_live.py`: la latencia en la PC como número principal y la de
      captura a resultado como referencia, con la nota del desfase de relojes. **Hecho
      y commiteado el 7/10**; el veredicto juzga la latencia en la PC y el del replay no
      cambia.
   1. EKF con el giroscopio, la velocidad como estado y el sesgo estimado, sobre las
      grabaciones completas con la compuerta: cortes de 10, 20 y 30 s, las condiciones
      (a) y (d), el lazo de 14:18 y los cuatro modos del OE4 (cámara sola, cámara +
      giroscopio, cámara + GPS, los tres). Tiene que ganarle al EKF actual en los cortes.
      **Hecho el 7/10 (§4.17), sin commitear:** le gana en todas las duraciones; el sesgo
      no ayudó y queda apagado. **Verificado el 8/10 (§4.18):** NIS y fixes absurdos; dos
      arreglos aprobados e implementados el mismo día: B (forzar un fix solo si el carro
      pudo llegar ahí desde el último fix que coincidió con la predicción) y C (el
      desajuste de la escala congelada en la incertidumbre durante los cortes). Ninguno
      cambia el trazo. **El EKF queda terminado**; al RL le falta su interfaz
      (innovaciones, NIS, incertidumbres, pesos ajustables), que va con la puerta del OE3.
   2. EKF en el pipeline: la IMU en replay (`gyro_accel.csv` en tiempo real) y en vivo
      (líneas `IMU` por el mismo socket UDP del GPS; sin ellas, el giro de la cámara),
      `GpsBuffer` con la velocidad Doppler, el estado fusionado en `frames.csv`, el modo
      solo visual y el simulador mandando líneas `IMU`. **Hecho el 8/10 (§4.19), sin
      commitear:** en determinista da lo mismo que `run_fusion` (0.5 mm); cuesta 0.14 ms
      por frame.
   3. El mapa en la ventana, junto al video: **ejes en metros, sin fondo de calles**
      (decisión del 5/10; mosaicos solo si sobra tiempo), con el GPS y la trayectoria
      fusionada, dibujado en tkinter. No depende de la app: se prueba con el replay y
      el simulador.
   **Paréntesis (8/10/2026): el estudiante pasa al task de la app antes del paso 3.**
   **Cerrado el 9/10/2026:** la app manda la IMU y responde a la consulta de hora; las
   dos pruebas con el teléfono real salieron bien (§4.20). La consulta de hora ya está
   en el pipeline. La latencia por cable quedó confirmada el 11/10 (p95 107 ms).
   Pendiente: que el filtro arranque y gire con el giroscopio (pide manejar). El orden era:
   - **Task de la app:** implementar el envío de la IMU (pedido 1 en "Pedidos pendientes
     para el task de la app", completo para pasarlo tal cual) y, si da tiempo, la
     consulta de hora (pedido 2).
   - **Al volver:** comprobar en vivo con el teléfono real, por cable, que la IMU llega y
     que el filtro la usa, siguiendo el "Cómo verificarlo" del pedido 1. La ventana lo
     tiene que indicar: antes hay que agregarle una línea con la IMU (frecuencia) y el
     estado de la fusión (esperando arranque, con giroscopio o con la cámara, σ), porque
     hoy no muestra nada de eso. La comprobación de que el filtro la usa pide manejar.
   - **Después: el paso 3**, el mapa en la ventana con el trazo fusionado.
   - **Después: la prueba de campo**; luego, la puerta del OE3 contra el EKF nuevo, el RL
     y el OE4.
4. **Hecho y verificado en esta sesión** (detalle en §4.9 y §4.14–4.16): EKF partes 1
   y 2; diagnóstico del lazo de la cámara sola; modo en vivo verificado con el
   simulador (equivale al replay); prueba por cable del estudiante (sin pérdidas; el
   objetivo de latencia, sin confirmar por el desfase de relojes); arreglos para
   trabajar sin internet; simulador del teléfono en el repositorio; zona UTM fija con
   el primer fix también en el replay (las posiciones actuales no cambian, verificado
   al metro); interfaz: `session.py` (`LiveSession`,
   `ReplaySession`), `run_live.py` como su consola (misma salida), `live_window.py` y el
   botón "SLAM GPS con realtime" en `interfaz.py` (lanzador a 800x600; el botón trae al
   frente la ventana si ya está abierta). La ventana se probó en Xvfb con datos
   simulados, límite de tiempo y conteo de ventanas: una sola en todo momento, video a
   ~13.5 imágenes/s en pantalla con el procesamiento a 30 Hz, en vivo con el simulador
   (grabando en el "teléfono") y en replay, detener, guardar y cerrar con la sesión en
   marcha. **Probada con el teléfono real por cable el 6/10** (§4.9): el video llega
   como en el teléfono; al terminar sin grabar en el teléfono apareció un bug que
   dejaba los botones bloqueados, ya corregido. Invariantes 6/6.
5. **Condiciones del estudiante para el EKF** (desde el 4/10): reportar siempre (a) el
   rumbo con y sin el curso y (d) cuántas correcciones se rechazan y cuántas se fuerzan.
   `evaluate.py --fusion` las imprime.
6. **La base de evaluación** son las grabaciones completas con la compuerta
   (`resultados/realtime/full_14_*_compuerta/`, 24 min, ~20 altos); los fragmentos de
   §4.13–4.14 se hicieron sin ella.
7. **Para la prueba de campo** (laptop sin internet, S23 por cable): chip en el S23 con
   los datos apagados y anclaje de red por USB; la IP del cable la muestra la app
   (fue 10.145.203.107). En la laptop, correr `setup.sh` una vez con internet: descarga
   el modelo y avisa si la GPU no sirve; después el pipeline carga el modelo sin
   consultar internet. Abrir la app al aire libre unos minutos antes (sin internet el
   primer fix tarda más); máscara 0.25, ajustable con la línea amarilla de la ventana.
   La primera carga del modelo después de encender la PC puede tardar más (34 s en la
   de escritorio, con disco mecánico, contra 3.8 s con la caché caliente). Desde el
   9/10 la latencia que muestra la ventana es la de captura a resultado, con el desfase
   de relojes medido en la sesión (§4.20), y entre paréntesis la de la PC. La app
   transmite la IMU (sin ella el EKF usaría el giro de la cámara, que en autopista hace
   el lazo). **Cómo dejar la laptop lista: §14; si algo falla allá: §15.**
8. Los scripts de análisis y las capturas de la ventana vivían en la carpeta temporal
   y se pierden; los números quedaron en §4.9 y §4.14–4.16.

### Ya hecho
- Escala por profundidad integrada al pipeline (+2.3 ms, n=3) e invariante T6.
- Puerta del OE3 sobre escala (§3.5.1), con la advertencia de que no se traslada al EKF.
- **App Android terminada** en otro task. Estado, protocolo y observaciones:
  `realtime/ESTADO_APP_ANDROID.md`.
- Alcance del mapa resuelto con el asesor (§11.1): trazar la trayectoria en vivo.
- **Cliente en vivo** (29/9/2026): `run_live.py` y las fuentes en vivo de
  `sources.py`, con precalentamiento. Probado con un simulador del protocolo, en
  el escritorio y caminando (§4.9).
- **Replay alineado por el reloj de arranque**, con soporte para las grabaciones de
  la app nueva (`--fy`, `--cx`, `--cy`, `--skip-start`). Números de referencia
  medidos de nuevo (§4.5).
- Validación de la VO contra el giroscopio (§4.10) y mapa del recorrido completo
  de `mobile_data/2025_03_11` (§4.8).
- **Dos grabaciones en carro** del 30/9/2026 (§8), traídas a `mobile_data/`.
- **Encadenamiento de poses corregido** (§4.11) y **`evaluate.py`** (2/10/2026).
- **Línea base solo visual medida** sobre un fragmento de cada grabación (§4.12).
- **Máscara en vez de recorte, empate de `recoverPose` corregido y red de seguridad
  de 10°** (§4.13), medidos sobre los dos fragmentos y la referencia vieja.
- **EKF, primera parte** (§4.14): el núcleo `[x, y, θ, s]` en `fusion.py`, evaluado
  sin conexión con cortes de GPS simulados.
- **EKF, segunda parte** (§4.15): compuerta de detenido en el estimador de
  profundidad, evaluada sobre las dos grabaciones completas.
- **Diagnóstico del lazo en autopista** (§4.16): es rumbo, no escala; la calibración no
  lo explica; el giro entre frames a 0.33 s lo quita. Decisión del 7/10: el rumbo sale
  del giroscopio.
- **Modo en vivo verificado con el código actual, prueba por cable y trabajo sin
  internet** (5/10/2026, §4.9).
- **Simulador del teléfono, zona UTM fija en el replay e interfaz** con video y modo
  replay (5/10/2026).

### Orden acordado
Primero la base solo visual sobre las grabaciones en carro; después la fusión,
desarrollada en replay sobre esas mismas grabaciones; la prueba de campo en vivo
al final, con todo junto.

| # | Tarea | Estimado |
|---|---|---|
| 1 | ~~Profundidad sobre la imagen completa (máscara en vez de recorte) y volver a medir la línea base~~ Hecho, más la corrección de los saltos (§4.13) | 1 día |
| 2 | EKF con la compuerta de detenido, en `realtime/` (núcleo y compuerta hechos, §4.14–4.15). Decidido el 7/10: rumbo del giroscopio, velocidad como estado y sesgo estimado; falta implementarlo y evaluarlo sobre las grabaciones (2–3 días) e integrarlo al pipeline con la IMU y el modo solo visual (2–3 días) | ~1 semana |
| 3 | ~~Interfaz tkinter~~ Hecha el 5/10/2026, sin el mapa. Falta el mapa en la ventana (ejes en metros) | ~1–2 días |
| 4 | Prueba de campo en vivo, por cable USB y sin internet, con la app transmitiendo la IMU | ~1 semana |
| 5 | Rehacer la puerta del OE3 contra el EKF fijo (incluye rumbo) | 3-4 días |
| 6 | RL, **con fecha límite**: cerrar a mediados de noviembre | ~2 semanas |
| 7 | Evaluación del OE4 (agregar el RPE) e informe, en paralelo desde noviembre | ~4 semanas |

Al 7/10/2026 quedan ~8–9 semanas hasta diciembre y la suma de lo que falta da ~8–9: el
giroscopio reemplaza al giro cada 10 frames y suma la IMU en vivo (~1 día más).
**Es alcanzable solo si el RL se acota por tiempo y se reporta lo que salga.**

### Notas para cada paso

**1. Profundidad sobre la imagen completa.** Hecho (§4.13). La línea base solo
visual del OE4 son ahora los resultados de `mask_vote_14_05/` y `mask_vote_14_18/`.

**2. EKF.** Núcleo hecho el 4/10/2026 (§4.14); lo que sigue vale para las partes que
faltan. Lo decidido en la sesión del 29/9/2026, con lo medido el 2/10/2026:
- **Estado `[x, y, θ, s]`.** `s` corrige la escala de la cámara y deja de ser
  opcional. El error de escala varía con la velocidad (la U de §3.3), así que `s`
  tiene que poder cambiar despacio. **Además depende del encuadre y de la escena
  (§4.13: de −30 % a +59 % según la grabación)**: `s` arranca con incertidumbre
  grande y se calibra con el Doppler desde los primeros segundos en movimiento.
- **Lo que el EKF recibe de la cámara ya viene limpio de saltos** (§4.13). Con la
  velocidad perfecta, el rumbo de la cámara sola separa 16–48 % en 4 min; en 1 s
  entre fixes deriva menos de 0.5°. El cabeceo (33–42° de deriva) no importa: el
  EKF usa solo el giro sobre la vertical. **Ojo:** sin IMU en el sistema, la
  vertical en ejes de la cámara no puede salir del acelerómetro (eso lo hace solo
  `evaluate.py`, como instrumento); hay que tomarla del montaje (eje y de la
  cámara) o estimarla de la propia VO. **Resuelto (§4.14):** el montaje está a ~3°
  de la gravedad y da el mismo rumbo que el acelerómetro. **Con el giroscopio en el
  sistema (7/10/2026)** la vertical sale del acelerómetro (2.1° y 4.6° del eje x del
  teléfono en las dos grabaciones).
- **Predicción** con cada frame: el giro de la VO sobre la vertical y el avance
  `s × paso`. **Corrección** con cada fix; la velocidad Doppler puede corregir `s`
  directamente. **Desde el 7/10/2026:** el giro sale del giroscopio (el de la VO si
  faltan muestras), con su sesgo como estado; la velocidad pasa a estado y la cámara
  queda como una medida de `v/s` que se puede rechazar.
- **El rumbo se corrige por la covarianza cruzada** entre rumbo y posición lateral,
  que solo crece en movimiento, **y con el curso entre fixes consecutivos** (§4.14). El rumbo inicial sale de los primeros fixes en
  movimiento, con incertidumbre grande (el marco de la cámara no coincide con el
  del mapa, §6.1).
- **La compuerta de detenido va con el EKF, no después.** Con el empate corregido,
  los altos ya casi no corrompen el rumbo (14:05: +15° en los altos; 14:18: 0°),
  pero la cámara sigue diciendo 0.6–0.9 m/s con el carro quieto: la compuerta
  tiene que frenar el avance y la escala (§4.10). **Hecha (§4.15):** la razón
  inliers/matches no sirvió; se usa el residuo tras el mejor giro puro entre frames
  a 0.33 s, dentro del estimador de profundidad.
- **Compuerta de giros imposibles: ya está en `VisualFrontEnd`** (10°, §4.13), como
  red de seguridad. La causa de los saltos se corrigió aparte.
- **Usar el encadenamiento correcto** (§4.11): giro `R^T`, avance `−R^T·t`.
- **El mapa que se muestra es el fusionado.** La cámara sola se separa 27–41 % de lo
  recorrido en 4 min (16–48 % aun con la velocidad perfecta); entre un fix y el
  siguiente (1 s) sí es útil.
- **El GPS en el carro va a 1 Hz** (§8): entre correcciones pasa un segundo.
- *Latencia del fix:* el `<t_ns>` del GPS y la marca de cada frame usan el mismo
  reloj, así que la corrección puede aplicarse en el instante correcto.
- *Atípicos:* compuerta sobre la innovación; el GPS del teléfono salta.
- `Q` y `R` fijas bien ajustadas son el baseline del OE3. La "regla de velocidad"
  de §3.5.1 se vuelve una `Q` programada por velocidad.
- La fusión tiene que poder desactivarse (modo **solo visual**) sin tocar el resto
  del pipeline: el OE4 compara los dos modos sobre la misma grabación.
- Para el OE4, reportar las dos formas: ATE con alineación (parecido de forma) y
  deriva desde un inicio común (más intuitiva).

**3. Mapa e interfaz.** **Interfaz hecha el 5/10/2026** (botón en `interfaz.py`,
ventana tkinter con el video, en vivo o de una grabación, y el simulador en el
repositorio). El video va dentro de tkinter, nunca en ventanas de OpenCV. Falta el
**mapa** junto al video: ejes en metros sin fondo de calles (decisión del
estudiante), con el GPS y la trayectoria fusionada (y, opcional, la cámara sola);
necesita el EKF en vivo. Toda prueba de ventanas, primero en Xvfb con el simulador,
con límite de tiempo y contando ventanas (reglas del `CLAUDE.md` global).

**4. Prueba de campo.** **Por cable USB** (anclaje de red), decidido tras la prueba
del 5/10 (§4.9): el punto de acceso queda descartado y no hace falta bajar la calidad
JPEG. Un recorrido cerrado da una medida de deriva gratis. Lista para trabajar sin
internet en "Dónde quedamos", punto 7.

**5 y 6. El RL, conceptualmente.** Elegir la fuente de escala es un *contextual
bandit*: la acción no cambia el estado siguiente, por eso una regla de un umbral
le ganó a un clasificador. Ajustar las covarianzas del EKF sí es secuencial (la
covarianza de ahora cambia la predicción e innovación siguientes): ahí el RL
tiene sentido. La recompensa debe mirar el error real contra el GT de KITTI.
Con ~10 minutos de datos, el entregable realista es implementación rigurosa con
techo medido. Al rehacer la puerta, usar como línea base de GPS la **velocidad
Doppler** (columna `speed`), que es mucho menos ruidosa que derivar posiciones:
el rival de la profundidad va a ser más fuerte que en §3.5.1.

### Pedidos pendientes para el task de la app
**Los pedidos 1 y 2 quedaron hechos el 9/10/2026** y verificados con el pipeline
(§4.20); se dejan como registro de lo que se pidió. Quedan los deseables.

El estudiante los pasa al task de la app tal cual. El primero es **firme** y es el
paréntesis del 8/10/2026; el segundo, si da tiempo; el resto, deseables. Lo del lado
de la app se revisó en su código (solo lectura).

**1. Transmitir la IMU por UDP (firme).**

*Para qué.* La PC estima el recorrido con un filtro de Kalman que combina la cámara,
el GPS y, desde el 7/10/2026 (aprobado por el asesor), el giroscopio del teléfono para
el rumbo. Con el giroscopio el rumbo deriva 1.2–1.7°/min en vez de 19–55°/min (el de
la cámara), los errores en los cortes de GPS bajan a la mitad (30 s sin GPS: 15.7
contra 31.6 m y 27.3 contra 44.4 m en las dos grabaciones de prueba) y el trazo sin GPS
deja de dar vueltas en la autopista (de 85 % a 18 % de lo recorrido). Hoy la app graba
la IMU en `gyro_accel.csv` pero no la transmite. **El lado de la PC ya está listo**:
lee las líneas `IMU`, las usa y se probó con un simulador que las manda. **Si no llegan,
el filtro sigue con el giro de la cámara**: funciona, pero sin GPS el rumbo deriva y en
autopista el trazo hace el lazo.

*Mensaje.* `IMU,<t_ns>,<gx>,<gy>,<gz>,<ax>,<ay>,<az>,<unix_ns>`: exactamente la fila de
`gyro_accel.csv` con el prefijo `IMU,` (nueve campos), como cada línea `GPS` es una fila
de `location.csv`. Es lo que ya da `SensorPacket.toString()` en `IMUManager`.
- Giroscopio en rad/s (`Sensor.TYPE_GYROSCOPE`, el calibrado) y acelerómetro en m/s²
  (`Sensor.TYPE_ACCELEROMETER`), en los ejes del dispositivo, con el acelerómetro
  interpolado al instante del giroscopio: lo que ya hace `syncInertialData()`.
- **Una muestra por datagrama**, como `send()` ya hace con cada línea: la PC lee un
  mensaje por datagrama, sin separar por saltos de línea.

*Frecuencia.* La de las grabaciones, ~53 Hz (`SENSOR_DELAY_GAME`, el valor por defecto
de la preferencia `prefImuFreq`): unos 90 bytes por línea, ~5 KB/s.

*Reloj.* `<t_ns>` es el `event.timestamp` del giroscopio, en
`SystemClock.elapsedRealtimeNanos()`: el mismo reloj que la marca de cada frame y el
`<t_ns>` del GPS, así que la PC alinea cada muestra con los frames sin desfase.
`<unix_ns>` como en el archivo (la hora Unix en ms seguida de `000000`).

*Cuándo.* Siempre que haya una PC suscrita, grabe o no, igual que el GPS. Los sensores
ya están registrados mientras la pantalla de video está abierta
(`CameraCaptureActivity.onResume` llama a `mImuManager.register()`).

*El cambio en la app.*
- `IMUManager`: un constructor que reciba el `SensorStreamer`, como `GPSManager(Activity,
  SensorStreamer)`, y que el actual `IMUManager(Activity)` siga existiendo con el
  streamer en `null`: lo usa `PhotoCaptureActivity`, que no transmite.
- En `IMUManager.onSensorChanged`, después de `syncInertialData()`, fuera de la condición
  `mRecordingInertialData`: si `syncedData != null` y hay streamer,
  `mStreamer.send("IMU," + syncedData.toString())`. La escritura al archivo queda como
  está. `send()` ya descarta la línea si no hay una PC suscrita: no hace falta
  comprobarlo antes.
- `CameraCaptureActivity`: crear el `IMUManager` con `mSensorStreamer`, como ya se hace
  con `new GPSManager(this, mSensorStreamer)`.
- `send()` no bloquea (encola para su propio hilo); la cola de 256 mensajes aguanta ~5 s
  si el envío se atrasa y, llena, descarta y lo cuenta.

*Documentación.* En `realtime/ESTADO_APP_ANDROID.md` (repositorio del pipeline, el
documento de traspaso de la app): la línea `IMU` en la tabla del §4.2; en el §2, "Sensores
por UDP" pasa a GPS e IMU; en el §5.5, la IMU "~53 Hz, en archivo y por UDP"; quitar del
§10 el punto "si la fusión necesita la IMU en vivo". En la app: la sección "Transmisión
en vivo" del `README.md`, y que `tools/udp_gps_receiver.py` reconozca las líneas `IMU`
(contarlas y mostrar su frecuencia una vez por segundo, en vez de imprimir "Línea no
reconocida" 53 veces por segundo).

*Cómo verificarlo.*
1. **Del lado de la app**, sin el pipeline: `tools/udp_gps_receiver.py` muestra ~53
   muestras de IMU por segundo, con `t_ns` creciente y del mismo orden que los `t_ns` de
   los frames y del GPS. Grabando, las filas recibidas coinciden con las de
   `gyro_accel.csv`.
2. **Con el pipeline, en casa** (sin manejar): una sesión en vivo de un minuto girando el
   teléfono con la mano.
   - La consola (`run_live`) termina con "Muestras de IMU recibidas: N (~53 Hz)" (hoy
     dice "ninguna").
   - La carpeta de la sesión trae `gyro_accel.csv`, y `frames.csv` trae la columna
     `gyro_yaw` llena.
   - `evaluate.py <sesión>/frames.csv --recording <sesión>` muestra, en "ROTACIÓN
     (contra el giroscopio)", el mapeo `[[0,-1,0],[-1,0,0],[0,0,-1]]` y un error por frame
     de ~0.2–0.3°, como con las grabaciones: ejes, unidades y reloj correctos. Necesita
     que la sesión tenga algún fix de GPS (cerca de una ventana alcanza): sin ninguno,
     `evaluate.py` hoy no corre.
   - La ventana muestra la frecuencia de la IMU. **Esa línea todavía no existe:** hoy la
     ventana no dice nada de la IMU ni de la fusión; se agrega en el task del pipeline
     antes de la comprobación.
3. **Que el filtro la usa** pide manejar: el filtro arranca cuando el GPS ve el carro
   avanzar 20 m a más de 3 m/s. Después del recorrido, la consola dice "Frames girados
   con el giroscopio: ~100 %" (`fusion.frames_con_giroscopio_pct` en `metrics.json`).
   Sin IMU dice "Sin muestras de IMU: el rumbo lo giró la cámara".

**2. Consulta de hora por UDP (si da tiempo)**, para medir la latencia sin el desfase de
relojes (§4.9; el desfase cambió al menos 154 ms entre el 29/9 y el 6/10). **Hecho; el
lado de la PC la usa desde el 9/10 (§4.20).**
- *Mensajes:* la PC manda `TIME,<pc_ns>` (su hora Unix en ns; cabe en el búfer de 64
  bytes del `listenLoop`). El teléfono responde de inmediato
  `TIME,<pc_ns>,<unix_ns>,<t_ns>`: el mismo `<pc_ns>`, su hora Unix con la misma
  conversión que usa `VideoStreamer` para los frames y `elapsedRealtimeNanos()`.
- Las dos horas del teléfono se toman al recibir el pedido, y la respuesta sale
  directo por el socket con `reply()`, como `SUBSCRIBED`, sin pasar por la cola de
  datos: la espera en la cola sesgaría la medida.
- Con la ida y vuelta, la PC estima la relación entre su reloj y el de arranque del
  teléfono con un error de la mitad de ese tiempo (pocos ms por cable). Usar el reloj
  de arranque evita los saltos de la hora del teléfono (sin datos ni WiFi la ajusta la
  red celular), y la latencia desde la captura vuelve a ser confiable.
- *Documentación:* el mensaje en el §4.2 de `ESTADO_APP_ANDROID.md` y en el `README.md`.

**Deseables:**
- Guardar el `HELLO` (o al menos cx y cy) en la carpeta de cada grabación.
- La precisión de cada fix (`getAccuracy`), como columna al final de `location.csv` y
  de la línea `GPS` (al final no rompe a los lectores actuales): una `R` realista para
  el EKF. El rumbo de cada fix (`getBearing`) pierde importancia con el giroscopio.
- Revisar por qué el GPS va a 1 Hz en el carro.

### Ideas medidas a medias o sin medir
- **Disparador por paralaje en `DepthScaleEstimator.try_update`.** Hoy dispara por
  tiempo (`min_baseline_s = 0.33`): a 1 m/s eso son 33 cm de base, y por eso el
  error a 0-2 m/s es 57 %. Esperar a una paralaje mínima en píxeles arreglaría el
  régimen lento sin sensores nuevos. Hipótesis sin medir; si funciona, le quita
  margen al RL.
- **Corrección de distorsión** (hasta ~17 px en las esquinas; coeficientes en §5.4
  de `ESTADO_APP_ANDROID.md`). La VO la ignora. **Medida el 4/10 (§4.16):** no cambia
  la deriva de rumbo (+53 contra +55°/min en la autopista; +20 contra +19 en ciudad).
  Podría valer por precisión, no por el lazo.
- ~~**Zona UTM.**~~ Resuelta el 5/10/2026: `sources.utm_zone` fija la zona con el
  primer fix en vivo, en el replay y en `evaluate.py` (el límite de 84° O pasa a 7 km
  de la casa del estudiante, hacia Cartago).
- **Más datos de KITTI.** Con las carpetas sin uso borradas (§8), cada secuencia
  nueva se puede bajar solo con `image_02` y `oxts`. Ojo: el GPS de KITTI es de
  grado navegación; el conflicto se simula con `degrade_gps` y hay que declararlo.
- **El sesgo de escala por grabación** (§4.13): con el mismo encuadre, 14:18 da 1.5
  veces lo de 14:05. Hipótesis sin medir: vehículos de adelante en el tránsito de
  14:05. Pista sin verificar: normalizar la profundidad por la focal efectiva a la
  entrada del modelo (la de este teléfono es 624 px; si el modelo asume ~1000 px,
  explica el ×1.6 de 14:18, no el de 14:05). Con dos grabaciones, ajustar una
  constante sería sobreajustar.
- **La deriva continua de rumbo de 14:05** (~20°/min) y los 12 pasos de 3–5° sin
  empate que el giroscopio no ve (§4.13). La distorsión y la focal quedaron
  descartadas; el giro entre frames a 0.33 s la baja a +4.5°/min (§4.16).
- **El sesgo de rumbo de la autopista de 14:18** (+30°/min aun con el giro a 0.33 s;
  cada 3 frames empeora). Candidatas sin medir: el tránsito que se mueve con el carro,
  el camino mojado. Hacen falta más datos de autopista (la prueba de campo). Con el
  rumbo del giroscopio, esta deriva y la de 14:05 solo importan para la línea base solo
  visual y para el respaldo sin IMU.
- **El cabeceo** ya se midió (33–42° contra el giroscopio, §4.13) con un script
  temporal; si hace falta reportarlo, agregarlo a `evaluate.py`.
- **Modo KITTI para el replay.** La verificación de §4.13 en KITTI se hizo con un
  script temporal. Una fuente de frames de KITTI (`image_02` + `oxts`) en
  `sources.py` permitiría correr el pipeline completo, y después el EKF, contra
  ground truth exacto. Útil para el OE4; no está pedido.
- **El estimador de profundidad** (`depth_scale.py:165`) usa `recoverPose` con el
  umbral por defecto. Se dejó así para medir un cambio a la vez; con 0.33 s de base
  empata menos, pero puede pasar a baja velocidad.
- **s en función de la velocidad** (§4.14): la escala del modelo de profundidad
  baja a la mitad sobre 14 m/s en 14:05. Aprender la pendiente con GPS y
  extrapolarla en un corte atacaría el error que domina los cortes. Con dos
  grabaciones (14:18 no pasa de 12 m/s) es fácil sobreajustar; KITTI tiene autopista.
- **Acortar la mediana móvil del estimador de profundidad** (7 estimaciones a 3 Hz,
  ~1 s de atraso, §4.14): el EKF ya promedia con s; menos atraso ayudaría al empezar
  un corte justo después de un cambio de velocidad. Hay que volver a correr el
  replay para medirlo.
- **Volver a verificar las tablas de trayectorias de §3.3** con el encadenamiento
  corregido (§4.11). La forma ya se verificó en KITTI con la distancia real
  (§4.13); falta con la escala del modelo de profundidad.

### Pendientes de limpieza (reglas del `CLAUDE.md` global)
- Hecho (27/9/2026): `scale_decision.py` y `depth_scale_probe.py` en inglés.
- Hecho (29/9/2026): el docstring de `run_replay.py` ya no menciona "OE1".
- Hecho (2/10/2026): este documento ya no usa emojis.
- `setup.sh` instala solo `opencv-python` desde el 5/10/2026; el entorno de este
  escritorio todavía tiene `opencv-python` y `opencv-contrib-python` a la vez.

### Estado de git al 11/10/2026
Desde el 11/10/2026 este documento y `realtime/CONTEXTO_APP_ANDROID.md` están en git,
para tenerlos en la laptop; también los 8 cachés de KITTI que leen los invariantes
(`resultados/rl_cache/*_cache.npz`, 392 KB). El resto de `resultados/`, `mobile_data/`
y `kitti_data/` sigue fuera.

### Estado de git al 9/10/2026
Último commit `579eb5a` (9/10/2026, `ESTADO_APP_ANDROID.md`; antes `516f2b1`, el paso 2,
`28a5e2e`, B y C, `6df67e4`, el paso 1, `fd6c771`, la consola, `f148de9`, la ventana, y
`14e90fc`, todo lo de la sesión del 4 al 6/10). Sin commitear, lo de §4.20; el mensaje
sugerido, en la respuesta del 9/10.
(Hasta el 11/10/2026 este documento estaba en `.gitignore`.)

---

## 11. PREGUNTAS ABIERTAS PARA EL PROFESOR ASESOR

1. ~~**ORB-SLAM vs odometría visual propia.**~~ **RESUELTO CON EL ASESOR
   (14/9/2026).** El alcance acordado es **trazar la trayectoria sobre un mapa
   en tiempo real**; no se exige mapa de landmarks, loop closure, bundle
   adjustment ni relocalización. Queda cerrado: no hay que implementarlos.

   *Lo que sí queda implicado:* lo que se traza debe ser la trayectoria
   **fusionada**, no la VO cruda (que deriva, §4.6). O sea, el mapa en vivo
   depende del EKF. Y hace falta la conversión **UTM → lat/lon**, que no existe
   en `utils/gps/gps_utils.py` (solo está `latlon_to_utm`); `pyproj` ya está
   instalado y la resuelve.
2. **¿Cómo se evalúa el OE3?** El objetivo dice "desarrollar e integrar" un
   mecanismo de RL — no dice que deba ganar. Conviene acordar si se evalúa como
   implementación rigurosa con techo medido.
3. ~~**¿Se mantiene la exclusión de IMU?**~~ **RESUELTO (7/10/2026): el asesor
   aprobó el giroscopio** para el rumbo (§4.16). Para el estudiante, el alcance
   declarado pesa menos que los objetivos específicos, que no cambian, y el margen
   menor del RL es aceptable. Los argumentos que se pesaron:
   - **A favor:** el giroscopio resolvería el rumbo en los altos, que es el error
     dominante (§4.8), y el acelerómetro detecta bien el vehículo detenido.
   - **En contra:** cambia el alcance declarado, exige que la app transmita la IMU
     (hoy solo la graba) y deja casi sin margen al RL, cuyo único caso genuino era
     justamente el rumbo en los altos (§3.5.1).
   - Usarla **solo como instrumento de validación**, fuera del sistema, no cambia el
     alcance (§4.10).
   - **Consultado por el estudiante el 5/10/2026; respuesta el 7/10: giroscopio**. Lo
     que se le llevó, medido el 4/10 (§4.16): el lazo de la cámara sola en autopista es de rumbo; con el
     giroscopio el trazo queda en 10–11 % de lo recorrido y sin lazo; sin IMU, estimar
     el giro entre frames a 0.33 s lo baja de 58–85 % a 14–34 %, también sin lazo
     (recomendado: no cambia el alcance).
4. **¿Es aceptable usar un modelo preentrenado de profundidad?** Es procesamiento
   visual con cámara monocular, así que cumple las limitaciones declaradas.

---

## 12. QUÉ VALE PARA LA TESIS

1. Caracterización de la VO sobre 8 secuencias con protocolo riguroso.
2. **El acoplamiento de escala**: el error de la cámara pasa de 41 a 250 m al
   degradarse el GPS, porque le pide prestada la escala.
3. **Descomposición de varianza**: dos tercios de la decisión de ponderación son
   ruido irreducible. Justifica cuantitativamente por qué ese planteamiento del RL
   tiene poco margen.
4. **Escala por profundidad monocular**: de dependiente a independiente del GPS,
   con validación en KITTI y en datos reales.
5. **El efecto de la base de triangulación**: 2.64x de sesgo a 1 frame, 1.08x a
   0.33 s. Resultado técnico transferible.
6. Estudio de viabilidad de escala por plano de tierra, con resultado negativo
   pero riguroso.
7. **Métricas de tiempo real**: 23 ms/frame, 45 ms p95, y el hallazgo de que el
   GPS está 20x más desactualizado que el cómputo.
8. **Un modelo de 88 ms cabe en un sistema de 30 Hz sin romperlo**: cuesta
   2.3 ms de latencia porque lo que produce (velocidad) es suave y se puede
   estimar a 3 Hz mientras la dirección sigue saliendo de la cámara a 30 Hz.
   Es un resultado de diseño de sistemas, no solo de visión.
9. Metodología de auditoría de fugas de ground truth y pruebas de invariantes.
10. **Validación de la VO con el giroscopio del teléfono, sin ground truth**
    (§4.10): mide el error de rotación en cualquier grabación de la app.
11. **El modo en vivo y el replay dan la misma precisión de rotación** (0.33° contra
    0.32°), así que las comparaciones del OE4 pueden hacerse en replay (§4.10).
12. **En vivo, el cuello de botella de la latencia es el enlace, no la PC**: la PC
    pone 25 ms de mediana; la cámara ~65 ms fijos y el WiFi los picos (§4.9).
13. **Un error de construcción heredado, encontrado midiendo**: el encadenamiento de
    poses al revés, escondido durante todo el proyecto por la alineación 3D (§4.11).
    Lección metodológica: evaluar en el plano horizontal y contra una referencia
    independiente (el giroscopio).
14. **Línea base solo visual en carro, medida contra GPS Doppler y giroscopio**
    (§4.12): el 0.4 % de los pasos explica la mayor parte del error de rotación, y
    el encuadre cambia la escala de un modelo de profundidad métrica.
15. **Un fallo de `recoverPose` con su configuración por defecto** (§4.13): con poco
    desplazamiento entre frames, la elección entre las cuatro soluciones empata y
    devuelve giros de 180° la mitad de las veces. Explica la "basura con el
    vehículo detenido" del sistema original; votar con todos los puntos lo corrige
    sin costo. Se encontró listando cada paso malo, no mirando promedios.
16. **La escala de un modelo de profundidad métrica depende de la escala de píxeles
    de su entrada** (§4.13): recortar la imagen cambia la profundidad un 30 %, y el
    contenido recortado casi no importa. Más un sesgo por escena (−30 % a +59 %):
    justifica que la escala se calibre en línea con el GPS en vez de tomarla como
    absoluta.
17. **Un EKF con los ruidos medidos, no ajustados** (§4.14): cada ruido sale de una
    medición contra el giroscopio o el Doppler, hecha antes de correr el filtro, y la
    consistencia se verifica con el NIS. En cortes de GPS simulados la cámara mantiene
    el trazo con ~17–30 % de lo recorrido y el error se descompone en escala y rumbo:
    dice qué sensor mejorar.
18. **Detectar el vehículo detenido solo con la cámara** (§4.15): la razón de inliers,
    el flujo frame a frame y el flujo crudo fallan, y se explica por qué (cuantización
    de ORB, el teléfono gira en los altos). El residuo tras el mejor giro puro entre
    frames a 0.33 s separa los dos casos, con un límite físico de velocidad para la
    escena lejana en autopista.
19. **El enlace decide la latencia en vivo** (§4.9): con el mismo teléfono y la misma
    PC, el cable USB no pierde frames y el WiFi pierde 7 %. El ancho p95 − p50 (5
    contra 88 ms) no depende del desfase de relojes, y los descartes de la PC por WiFi
    se explican por la llegada en ráfagas. El p95 absoluto (113 contra 212 ms) sí
    depende, y ese desfase cambia de un día a otro: para reportarlo hay que medirlo
    en la misma sesión.
20. **Separar el error de la cámara sola en rumbo y escala con instrumentos** (§4.16):
    combinando el giro de la cámara o del giroscopio con la velocidad de la cámara o
    del Doppler se ve qué produce el lazo (el rumbo), y que la calibración no lo
    explica. Mismo método que §4.13, aplicado a una pregunta de diseño.
21. **La regla fija que sirve en una sesión empeora otra** (§4.27): cuándo dejar de
    confiar en la velocidad de la cámara durante un corte de GPS. Hay margen en todas
    las sesiones (oráculo, 30 s: p90 −32 a −57 %), pero el umbral de velocidad que lo
    captura cambia con la sesión (7–13 m/s): ajustado en 14:05 y probado en 14:18
    captura el 46–86 % del margen; al revés, empeora al EKF (−32 a −115 %). Es el
    argumento medido para una política que mira varias señales, y la razón por la que
    toda comparación (RL y reglas) va dejando una sesión afuera.

---

## 13. CÓMO RETOMAR EN UNA SESIÓN NUEVA

1. Leer §0, §2 y §10 de este documento. Si el trabajo toca la conexión con el
   teléfono, leer también `realtime/ESTADO_APP_ANDROID.md`.
2. Correr `venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.sanity_checks` — los 6
   invariantes deben pasar.
3. Correr `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay mobile_data/2025_03_11 --frames 900`
   para verificar el pipeline (debe dar ~24–27 ms/frame).
4. Con escala métrica, agregar `--scale depth --trajectory`
   (**desde el 6/10/2026: 255 m contra 189 m del GPS, RMSE 17.45 m, mediana 18.22 m, 1
   pose descartada, 8 estimaciones de escala en alto**; el cambio lo trajo la compuerta
   de detenido de §4.15, verificado por commits el 11/10. Del 2/10 al 4/10, con la
   máscara y la corrección de §4.13: 254 m, RMSE 17.29 m, mediana 17.85 m; sin escala,
   RMSE 21.61 m y mediana 20.93 m. Los de antes, 204 m y 16.59/11.37 m, eran con el
   recorte; historia en §4.5 y §4.13). El modo estricto se pide con `--strict`.
5. Las grabaciones de la app nueva necesitan los intrínsecos del `HELLO` y
   `--skip-start 6` (comando en §8). El modo en vivo se corre con
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live <IP del teléfono>`;
   la IP cambia, se lee en la pantalla de video de la app.
6. La ventana del pipeline se abre con `venv/bin/python interfaz.py` → botón
   "SLAM GPS con realtime", o sola con
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.live_window`. Recuerda la última
   IP, grabación y opciones en `resultados/realtime/interfaz.json`.
7. Sin el teléfono, el modo en vivo se prueba con el simulador:
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_simulator mobile_data/<carpeta> --start 380 --duration 120`
   y, en otra terminal, `run_live 127.0.0.1`. El simulador manda la IMU de la grabación
   (`--no-imu` para imitar la app actual); la fusión corre siempre (`--visual-only` y
   `--no-gyro` en `run_live` y `run_replay`).
8. Para evaluar una sesión contra el GPS y el giroscopio de su grabación:
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.evaluate <frames.csv> --recording mobile_data/<carpeta>`
   (los comandos de los fragmentos de la línea base están en §4.12, con
   `--mask-bottom` en lugar de `--crop-bottom`). Con `--fusion` evalúa también el
   EKF, y `--outage 120 30` simula un corte de GPS de 30 s en el segundo 120 (§4.14);
   con `--sweep` compara las variantes del filtro sobre un corte cada 10 s (§4.17). Las
   grabaciones completas con la compuerta están en `resultados/realtime/full_14_*_compuerta/`.
9. En una laptop nueva: §14 (pasos exactos). Antes de una prueba de campo, repasar
   la lista de "Dónde quedamos", punto 7. Si algo falla: §15.
10. Verificar con `git status` qué cambió desde entonces.
11. **Continuar por §10, apartado "Dónde quedamos".**

**Principios que funcionaron y conviene mantener:**
- **Medir antes de construir.** Cada etapa con puerta de decisión numérica explícita.
- Incrementos pequeños y verificables.
- **Auditar fugas de ground truth siempre.** Varias conclusiones cambiaron al hacerlo.
- Ante un resultado sorprendentemente bueno o malo, **sospechar del código propio primero**.
- **No declarar éxito con una sola secuencia.** Pasó dos veces en esta sesión.
- **En un modo no determinista, n=1 no mide nada.** Para reportar una diferencia
  de milisegundos hay que correr ≥3 veces cada configuración y comparar rangos.
- **Un invariante que nunca falla no vale nada.** Probar que falla cuando debe.

---

## 14. CÓMO DEJAR EL AMBIENTE LISTO EN OTRA COMPUTADORA

Para la laptop de las pruebas de campo (Linux Mint 22 con GPU NVIDIA). Se hace en casa,
con internet; en campo el sistema no necesita internet.

**Lo que hace falta:** Linux Mint 22 o Ubuntu 24.04 (traen Python 3.12; Mint 21 no),
driver NVIDIA 560 o más nuevo, ~12 GB libres (el entorno virtual ocupa 7.2 GB y la caché
de `pip` ~4 GB durante la instalación) más los datos que se copien, y acceso a GitHub.

### Pasos

1. **Driver de la GPU.** `nvidia-smi` tiene que mostrar la tarjeta y una versión de
   driver ≥ 560. Si no, instalarlo desde el Administrador de controladores de Mint
   (el recomendado) y reiniciar. Con Secure Boot activo, Mint pide registrar la clave
   del módulo al reiniciar; si no se hace, el driver no carga.
2. **Clonar el repositorio** en la misma ruta que en la PC de escritorio (las rutas no
   importan para el código, pero así los comandos de este documento sirven tal cual):
   ```
   mkdir -p ~/Documents/Github && cd ~/Documents/Github
   git clone https://github.com/NJohnsonCR/Slam_GPS_Design.git
   cd Slam_GPS_Design
   git checkout SLAM_RealTime
   ```
3. **Correr `setup.sh`**, con internet: `./setup.sh`. Pide la contraseña para `apt`
   (Python 3.12, tkinter, `adb`), crea `venv/` con las versiones fijas de
   `requirements.txt` (un solo OpenCV), descarga el modelo de profundidad (~100 MB, a
   `~/.cache/huggingface`) y revisa la GPU. Tiene que terminar con
   `GPU disponible: <nombre de la tarjeta>` y `Entorno configurado correctamente.` Si
   dice `AVISO`, ver §15. Se puede volver a correr sin problema.
4. **Copiar los datos** que no están en git, desde la PC de escritorio (memoria USB o
   `rsync` por la red). Para las pruebas de la laptop alcanzan dos grabaciones, ~1.2 GB:
   ```
   rsync -av --progress mobile_data/2025_03_11 mobile_data/2026_09_30_14_18_36 \
       <usuario>@<ip de la laptop>:~/Documents/Github/Slam_GPS_Design/mobile_data/
   ```
   `kitti_data/` (2.6 GB) no hace falta: los invariantes leen los cachés que están en
   git.
5. **Verificar** (los mismos números que en la de escritorio, §14 "Verificación"):
   ```
   venv/bin/python -m LMS.LMS_RL_ORB_GPS.scripts.rl.sanity_checks
   venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay mobile_data/2025_03_11 --frames 900
   venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay mobile_data/2025_03_11 --frames 900 --scale depth --trajectory
   ```
   - Invariantes 6/6.
   - 900 frames sin descarte; el tiempo por frame depende de la CPU (24–27 ms en la de
     escritorio).
   - Con escala: `Modelo listo, en la GPU ...`, 255 m, RMSE 17.45 m, mediana 18.22 m.
     Tienen que salir **iguales**: el modo determinista es reproducible; si cambian,
     algo del ambiente es distinto.
6. **Sin internet** (como en campo): desconectar la red y repetir el último comando.
   Tiene que cargar el modelo igual, sin esperar.
7. **El modo en vivo, sin el teléfono:** en una terminal
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_simulator mobile_data/2026_09_30_14_18_36 --start 380 --duration 120`;
   en otra, `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.live_window`, IP `127.0.0.1`,
   Iniciar. El video corre, `IMU: 53 muestras/s; el giro lo da el giroscopio`,
   `Fusión: en marcha` a los pocos segundos, `latencia ~35 ms (en la PC ~30)`.
8. **Con el teléfono:** la app en la pantalla de video, en la misma red (o por cable con
   anclaje USB, la IP la muestra la app). Primero
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_check <IP>`: tiene que decir
   `Todo llega, en el mismo reloj.` Después la ventana con esa IP.

### Verificación en la PC de escritorio antes de pasar a la laptop (11/10/2026)

- **`setup.sh` desde cero**, en un clon limpio, con una caché de modelos vacía: instaló
  las versiones de `requirements.txt`, un solo OpenCV (4.12.0.88), bajó el modelo y
  encontró la GPU (`torch 2.8.0+cu128`, RTX 3070). Corrido dos veces seguidas en el
  mismo clon: la segunda solo completó lo que faltaba. El paso de `apt` se probó en
  simulación (`apt-get install -s`): los 8 paquetes se resuelven en Mint 22.1
  (`python3-tk` y `libglib2.0-0t64` son los nombres reales en Ubuntu 24.04).
  **La primera corrida se quedó sin espacio** en la de escritorio (11 GB libres): con
  ~12 GB alcanza.
- Con ese entorno nuevo y `HF_HUB_OFFLINE=1`: el replay con escala dio lo mismo que el
  entorno de siempre (255 m, RMSE 17.45 m, 24.6 ms por frame).
- Invariantes 6/6, también en un clon con solo el código y los cachés de git.
- Replay de 900 frames: 0 descartados, 27.0 ms por frame (p50; con la instalación de
  prueba corriendo al lado), fusión con el giroscopio desde el segundo 9.
- Con escala y trayectoria: los números de arriba, idénticos en dos corridas.
- La ventana con el simulador en Xvfb (en vivo con IMU, con la IMU cortada, replay):
  todo bien, la consulta de hora con `reloj.csv` en las sesiones en vivo.
- `phone_check.py` con el simulador, con una IP sin teléfono y con el teléfono sin la
  app abierta: cada caso da su mensaje y su código de salida.
- **El teléfono real** (§4.20): por WiFi, los tres sensores llegan en el mismo reloj
  (`phone_check`: todo bien; `run_live` 30 s: IMU 53 Hz, giroscopio en el 99.9 % de los
  frames). Por cable con anclaje USB, 60 s: 0 frames perdidos, captura a resultado p95
  107 ms, cumple.

---

### Verificación en la laptop (11/10/2026)

- `setup.sh` instaló `torch 2.8.0+cu128`, que ve la GTX 1050 Ti pero no corre en ella
  (sm_61). Cambiado a `+cu126` (los mismos 2.8.0, con sm_50–sm_90); `requirements.txt`
  lo fija desde entonces. `torch.cuda.is_available()` True, `Modelo listo, en la GPU
  NVIDIA GeForce GTX 1050 Ti`, `pip check` limpio.
- Invariantes 6/6.
- Sin `mobile_data/` en la laptop: la ventana se probó con el simulador sirviendo una
  grabación sintética (calle con textura a 8 m/s, GPS 1 Hz, IMU 53 Hz; generada fuera
  del repo), 30 s: `IMU: 53 muestras/s; el giro lo da el giroscopio` y `Fusión: en
  marcha` desde el segundo 9; al final el giroscopio giró el 100 % de los frames.
- Con el teléfono por cable: `phone_check` bien y dos sesiones en vivo; la latencia
  p95 no cumple en esta laptop (§4.20).
- **Pendiente:** los números del paso 5 con `mobile_data/2025_03_11`, y el trazo con
  descarte real (`--strict`, §4.20).

---

## 15. SI ALGO FALLA EN LA LAPTOP

Cada comando, desde la raíz del repositorio.

### La GPU

- `nvidia-smi`: si no responde, el driver no está cargado (instalarlo o, con Secure
  Boot, registrar la clave; §14 paso 1).
- `venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`:
  - `False`: driver ausente o más viejo que lo que pide CUDA 12.6 (≥ 560).
  - La versión tiene que terminar en `+cu126`. `requirements.txt` la fija así desde el
    11/10/2026: la de PyPI (`+cu128`) no trae código para tarjetas anteriores a
    (7, 0), como la GTX 1050 Ti de la laptop (6, 1), y `setup.sh` avisaba "hay GPU,
    pero esta versión de torch no corre en ella". Si un entorno viejo todavía tiene
    `+cu128`, `./setup.sh --sin-apt` lo cambia.
- Sin GPU el sistema funciona igual, más lento en la escala: en la CPU de escritorio
  una inferencia tardó ~615 ms (§4.9), así que la escala se actualiza a ~1.5 Hz en vez
  de 3 Hz y le quita CPU al resto. El procesamiento por frame no usa la GPU.

### El modelo de profundidad

- Está en `~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Metric-Outdoor-Small-hf/`.
  Si falta, correr `./setup.sh` con internet (o solo su paso del modelo).
- Probarlo sin internet:
  `HF_HUB_OFFLINE=1 venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_replay mobile_data/2025_03_11 --frames 300 --scale depth`
  tiene que decir `Modelo listo, en la GPU ...` (o en la CPU).
- La primera carga después de encender puede tardar ~30 s con disco lento; las
  siguientes, ~4 s.

### El teléfono y los sensores

1. **¿Se ve?** `ping <IP>` (la IP la muestra la pantalla de video de la app). Por cable,
   `ip -4 addr` tiene que mostrar una interfaz nueva (`usb0` o `enx...`) en la misma red.
2. **¿Manda?** `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.phone_check <IP>`, sin el
   pipeline:
   - `Video : NO se pudo conectar (... refused)`: la app no está en la pantalla de video.
   - `UDP : el teléfono NO responde`: lo mismo, o el puerto 5001 cambiado en los ajustes.
   - `GPS : 0 fixes`: bajo techo tarda; al aire libre, unos minutos tras abrir la app.
   - `IMU : NO llega` o `Hora : NO responde a TIME`: la app es una versión vieja.
   - `Reloj : ... NO cuadra`: frames e IMU en relojes distintos; problema de la app.
   Si `phone_check` da todo bien y el pipeline falla, el problema es del pipeline; si
   `phone_check` falla, es de la app o de la red (y va al task de la app).
3. **El pipeline en vivo:**
   `venv/bin/python -m LMS.LMS_RL_ORB_GPS.realtime.run_live <IP> --duration 30`: la
   línea de cada segundo dice `giro: giroscopio (53 Hz)`, la latencia con `(PC ..)`, y
   al final `Muestras de IMU recibidas: ... (53 Hz)` y el desfase de relojes medido.
4. **Traer una grabación del teléfono** (con `--record` o el botón de la app):
   `adb devices` (el teléfono tiene que aparecer como `device`; si dice `unauthorized`,
   aceptar en el teléfono) y
   `adb pull /sdcard/Android/data/edu.osu.pcv.marslogger/files/data/<carpeta> mobile_data/`.
   La carpeta la imprime `run_live` al terminar (`Carpeta de la grabación en el teléfono`).

### El ambiente de Python

- `venv/bin/pip list | grep -i opencv`: tiene que haber **una** línea
  (`opencv-python`). Si hay dos, `./setup.sh` lo arregla.
- `venv/bin/pip check`: sin dependencias rotas.
- La ventana no abre (`No module named tkinter` o `ImageTk`): falta `python3-tk`
  (`sudo apt install python3-tk`).
- Los invariantes no corren por `FileNotFoundError` en `resultados/rl_cache`: falta el
  `git pull` que trae los cachés.
