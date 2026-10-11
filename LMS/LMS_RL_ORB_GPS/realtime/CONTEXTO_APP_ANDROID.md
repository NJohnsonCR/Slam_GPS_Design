# Contexto para el desarrollo de la app Android

**Para:** una sesión de trabajo dedicada exclusivamente a la app móvil.
**Última actualización:** 8 de septiembre de 2026
**Documento hermano:** `scripts/rl/CONTEXTO.md` (estado del lado PC)

Este documento es **autocontenido**. No hace falta conocer el proyecto de SLAM
para trabajar con él.

---

## 0. RESUMEN EN 8 LÍNEAS

1. Se necesita una app Android que capture **video + GPS** (IMU opcional) y los **transmita en vivo por WiFi** a una PC que corre un sistema de localización visual.
2. La PC ya está construida, probada y midiendo: procesa a 30 Hz con 45 ms de latencia.
3. La base recomendada es **MARS Logger** (`github.com/OSUPCVLab/mobile-ar-sensor-logger`), que ya resuelve lo difícil: sincronizar sensores contra un solo reloj.
4. **Le falta el GPS** (está en su roadmap, no implementado) y **le falta transmitir** (solo graba).
5. **Primera tarea, antes que nada: medir a qué frecuencia real entrega GPS el teléfono.** Se venía asumiendo 1 Hz, pero probablemente sea configurable. Ese número condiciona el diseño del lado PC.
6. Protocolo definido: **TCP para video, UDP para sensores**, todo estampado con el reloj monótono del dispositivo.
7. La app debe **grabar y transmitir simultáneamente**, nunca una sola cosa.
8. **No fusionar con otras apps** (ver §7). El trabajo es agregar dos módulos a MARS Logger, no combinar proyectos.

---

## 1. PARA QUÉ SIRVE

Alimenta un sistema de SLAM visual que corre en una PC con Linux/Python. La PC
recibe frames y posiciones GPS, y estima la trayectoria del vehículo en tiempo
real durante recorridos controlados en carretera.

**Restricciones que impone el sistema receptor:**

| Requisito | Valor | Por qué |
|---|---|---|
| Latencia extremo a extremo | < 150 ms (p95) | Objetivo de tiempo real del proyecto |
| Tasa de video | 30 fps | La PC procesa cada frame en ~23 ms |
| Precisión de timestamps | ~10 ms | El GPS llega cada 500-1000 ms; no hace falta más |
| Formato de archivos | El de MARS Logger | Para que el modo replay de la PC funcione igual |

Notar que la precisión de sincronización requerida es **mucho más floja** que la
que MARS Logger fue diseñado para dar (nanosegundos entre cámara e IMU). No hace
falta esa exactitud, pero tampoco estorba.

---

## 2. QUÉ ES MARS LOGGER Y QUÉ LE FALTA

Proyecto académico (Ohio State, paper 2020). **Android en Java**, licencia
**GPL-3.0**. Forkear y modificar es legal; si se distribuye el binario hay que
publicar las modificaciones.

**Ya hace, no reimplementar:**
- Captura video H.264/MP4 vía Camera2
- Metadata por frame: intrínsecos, exposición, ISO, skew del obturador
- IMU (giroscopio + acelerómetro) a ~100 Hz
- **Sincroniza todo contra un solo reloj** ← su aporte central
- Escribe a archivos CSV/TXT

**No tiene:**
- **GPS** (roadmap, no implementado)
- **Transmisión** (solo graba a disco)

---

## 3. ARQUITECTURA PROPUESTA

```
┌─────────────────── TELÉFONO ───────────────────┐        ┌──── PC ────┐
│                                                 │        │            │
│  Camera2 ──┬──> escritor de archivos (MP4+CSV)  │        │            │
│            │                                    │        │            │
│            └──> codificador JPEG ──> TCP :5000 ─┼───────>│  video     │
│                                      (descarta  │        │            │
│                                       si ocupado)│        │            │
│  GPS ──────┬──> location.csv                    │        │            │
│            └──> UDP :5001 ──────────────────────┼───────>│  sensores  │
│                                                 │        │            │
│  IMU ──────┬──> gyro_accel.csv                  │        │            │
│  (opcional)└──> UDP :5001                       │        │            │
│                                                 │        │            │
│  Reloj: SystemClock.elapsedRealtimeNanos()      │        │            │
│         para TODOS los sensores                 │        │            │
└─────────────────────────────────────────────────┘        └────────────┘
```

**El teléfono actúa como servidor:** escucha, muestra su IP en pantalla, y la PC
se conecta. Razón práctica: con el hotspot del teléfono su dirección es
predecible, y se evita tener que escribir la IP de la PC en el celular.

---

## 4. PROTOCOLO

### 4.1 Saludo (al conectarse la PC, por TCP)

```
HELLO,<ancho>,<alto>,<fx>,<fy>,<cx>,<cy>,<fps>\n
```

**Esto es importante:** resuelve el problema de la calibración de raíz. La PC
aprende los parámetros de cámara al conectarse en vez de tenerlos escritos en el
código — que fue exactamente un bug encontrado en el sistema (usaba la
calibración de KITTI para video de teléfono, con 59 % más de error).

### 4.2 Video (TCP, puerto 5000)

Cada frame:

| Campo | Bytes | Tipo |
|---|---|---|
| longitud del payload | 4 | uint32 big-endian |
| timestamp monótono (ns) | 8 | uint64 |
| tiempo Unix (ns) | 8 | uint64 |
| JPEG | N | — |

**JPEG, no H.264.** H.264 es más eficiente pero obliga a manejar unidades NAL,
SPS/PPS y un decodificador del lado PC. JPEG a 720p son 100–200 KB por frame
(3–6 MB/s a 30 fps), manejable en WiFi, y del lado de Python es una línea
(`cv2.imdecode`). Optimizar solo si el ancho de banda resulta un problema medido.

**Regla no negociable:** escritura **no bloqueante**. Si el socket no está listo,
**descartar el frame**. Nunca encolar. Si se encola, la latencia crece sin límite
y el sistema deja de ser tiempo real aunque el contador de fps se vea bien.

### 4.3 Sensores (UDP, puerto 5001)

Un datagrama por medición, texto plano, **el mismo formato que las filas del CSV**:

```
GPS,<t_ns>,<lat>,<lon>,<alt>,<speed>,<unix_ns>
IMU,<t_ns>,<gx>,<gy>,<gz>,<ax>,<ay>,<az>,<unix_ns>
```

Que el stream lleve el mismo formato que el archivo permite que la PC use **un
solo parser** sin importar la fuente.

---

## 5. SINCRONIZACIÓN — LA PARTE CRÍTICA

Todo se estampa con **`SystemClock.elapsedRealtimeNanos()`**, el reloj monótono
del dispositivo. Dos detalles que, si se hacen mal, rompen la sincronización sin
que se note:

**Cámara:** usar el timestamp del *sensor* (`CaptureResult.SENSOR_TIMESTAMP`), no
el momento en que la app recibió el frame. El archivo `edge_epochs.txt` de las
grabaciones existentes confirma que en este dispositivo
`SENSOR_INFO_TIMESTAMP_SOURCE` es `SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME`, o sea
que ese timestamp es **directamente comparable** con `elapsedRealtimeNanos()`.

**GPS:** usar `Location.getElapsedRealtimeNanos()`, **no** `getTime()`. El primero
está en el reloj monótono; el segundo es hora de pared y no sirve para alinear
con la cámara.

El campo `Unix time` se incluye además como referencia absoluta, para poder
correlacionar con otros dispositivos o revisar grabaciones después.

---

## 6. FORMATO DE ARCHIVOS OBJETIVO

Existe una versión modificada de MARS Logger que **ya tenía GPS** y produjo los
datos con los que la PC fue desarrollada. No se tiene su código, pero **sí sus
archivos de salida**, así que el formato a reproducir es conocido exactamente.

Muestras en `mobile_data/2025_03_11/` del repositorio de SLAM.

### `location.csv` — hay que crearlo
```
Timestamp[nanosecond],latitude[degrees],longitude[degrees],altitude[meters],speed[meters/second],Unix time[nanosecond]
42220951807625,9.88675154,-84.07126274,1193.0460205078125,0.0,1762119716038000000
```

### `movie_metadata.csv` — existe en upstream, **falta la última columna**
```
Timestamp[nanosec],fx[px],fy[px],Frame No.,Exposure time[nanosec],Sensor frame duration[nanosec],Frame readout time[nanosec],ISO,Focal length,Focus distance,AF mode,Unix time[nanosec]
```

### `gyro_accel.csv` — existe en upstream, **falta la última columna**
```
Timestamp[nanosec],gx[rad/s],gy[rad/s],gz[rad/s],ax[m/s^2],ay[m/s^2],az[m/s^2],Unix time[nanosec]
```

### `frame_timestamps.txt`
```
Frame timestamp[nanosec],Unix time[nanosec]
```

---

## 7. FUNCIONALIDAD MÍNIMA

### Imprescindible
- [ ] Captura de video a 30 fps, **orientación bloqueada en horizontal**
- [ ] GPS a la máxima frecuencia que entregue el dispositivo
- [ ] Todo estampado con el reloj monótono (§5)
- [ ] **Grabar a archivos** en el formato de §6
- [ ] **Transmitir** video (TCP) y GPS (UDP) según §4
- [ ] Pantalla de configuración: puertos, calidad JPEG, resolución
- [ ] Mostrar la **IP del dispositivo** en pantalla
- [ ] Botón iniciar / detener
- [ ] Estado en vivo: fps enviados, tasa de GPS, estado de conexión, **frames descartados**

### Deseable, después
- [ ] IMU por el mismo canal UDP
- [ ] Reconexión automática si se cae la red
- [ ] Selección de cámara (usar la **gran angular** si existe: amplía el campo de visión, que favorece la estimación de profundidad del lado PC)

### Explícitamente fuera de alcance
- Interfaz elaborada, cuentas de usuario, nube, H.264 en la primera versión

---

## 8. FASES

### Fase 0 — Compilar sin modificar (2–3 días) ⟵ EMPEZAR AQUÍ
Android Studio, clonar MARS Logger, compilar **tal cual**, correr en el teléfono.

**Puerta:** genera `movie.mp4`, `movie_metadata.csv`, `gyro_accel.csv` y
`frame_timestamps.txt`.

> **Riesgo principal del proyecto.** Código de ~2020; es probable que haya que
> actualizar Gradle, el Android Gradle Plugin y el target SDK. **Si se traba más
> de 3 días, replantear**: quizá partir de un proyecto nuevo con CameraX salga
> más barato que resucitar el viejo.

### Fase 0.5 — MEDIR LA FRECUENCIA DE GPS (medio día) ⟵ CRÍTICO
Agregar un listener pidiendo **10 Hz** y medir **qué entrega realmente**.

Se venía asumiendo 1 Hz porque es lo que dieron las grabaciones existentes, pero
ese valor probablemente sea el default de la app original, no un límite del
hardware. Muchos chips GNSS entregan 5–10 Hz si se los pide.

**Reportar ese número al lado PC.** Cambia decisiones de diseño: si son 5 Hz, el
hueco entre correcciones se reduce cinco veces.

### Fase 1 — GPS a archivo (3–4 días)
Permiso `ACCESS_FINE_LOCATION`, `FusedLocationProviderClient`, escribir
`location.csv` con el formato exacto de §6 y el reloj de §5.

**Puerta:** grabar 2 minutos y comparar contra `mobile_data/2025_03_11/location.csv`.

### Fase 2 — Transmitir sensores por UDP (2–3 días)
`DatagramSocket` + hilo emisor con cola. Donde escribe cada fila del CSV,
además encolar un datagrama.

**Puerta:** un script de Python de 15 líneas en la PC recibe y muestra las líneas.

### Fase 3 — Transmitir video por TCP (≈1 semana)
Salida JPEG en paralelo a la grabación, con el encabezado de §4.2 y **descarte si
el socket no está listo**.

> Detalle técnico: MARS Logger usa **Camera2** con una sesión de captura ya
> configurada. Agregar salida JPEG implica añadir un `ImageReader`/`Surface` más
> a esa sesión. Es la parte más delicada de la app.

**Puerta:** la PC muestra el video en vivo; medir latencia y ancho de banda.

### Fase 4 — Interruptor y campo (≈1 semana)
Toggle grabar / transmitir / ambas. **Mantener siempre la grabación.** Pruebas en
vehículo con hotspot si el WiFi es inestable.

**Total estimado para alguien nuevo en Android: ~3 semanas.**

---

## 9. QUÉ **NO** HACER

**No fusionar con `GPS-Video-Logger`** (`github.com/abinpaul1/GPS-Video-Logger`).
Se evaluó y descartó:
- Guarda el GPS en **GPX**, con timestamps ISO 8601 de resolución de segundo. Se
  necesitan nanosegundos en el reloj monótono.
- Su sincronización es "los archivos se llaman igual", a nivel de sesión.
- No maneja IMU.
- Fusionar dos apps Android cuesta más que agregar un listener de ubicación.

**Sí sirve como referencia de código** para permisos en runtime y ciclo de vida
del proveedor de ubicación. Es MIT, compatible con GPL-3.0.

**No reemplazar la grabación por el streaming.** Conviven.

**No encolar frames cuando la red se atrasa.** Descartar.

---

## 10. DECISIÓN PENDIENTE: MARS Logger vs Flutter

Se discutió y quedó **sin resolver**. Depende de si se necesita la IMU:

| | MARS Logger (Java, nativo) | Flutter (Dart, de cero) |
|---|---|---|
| Tiempo a primera versión | ~3 semanas | ~1.5–2 semanas |
| Riesgo principal | Que no compile (proyecto de 2020) | Rendimiento de `startImageStream` |
| Sincronización | Excelente | Suficiente para el caso |
| Metadata por frame | Sí | No |
| Sirve para iOS | No | Sí |

**La regla de decisión:** si la escala del sistema termina dependiendo de la
**IMU**, hace falta sincronización precisa → **MARS Logger**. Si no (que es el
caso actual: la escala sale de un modelo de profundidad, ver documento hermano),
**Flutter es viable y más rápido**.

**Estado actual: la escala NO depende de la IMU**, así que Flutter está sobre la
mesa. Decisión del estudiante.

---

## 11. DATOS DEL DISPOSITIVO Y LA GRABACIÓN EXISTENTE

De `mobile_data/2025_03_11/`:

| Propiedad | Valor |
|---|---|
| Video | 720×1280 **vertical**, 29.97 fps |
| Distancia focal | fx ≈ 899 px, muy estable (897.9–907.6, desv. 1.51) |
| Obturador rodante | 20.8 ms de lectura por frame |
| GPS | 1.00 Hz exacto — **verificar si es configurable** |
| IMU | 49.9 Hz |
| Duración | ~289 s |

**Observaciones del montaje que conviene mejorar:** el tablero del vehículo ocupa
la parte baja del cuadro (la PC lo recorta un 14 %), la mitad superior es cielo
desaprovechado, y hay desenfoque de movimiento notable.

**Recomendaciones para futuras grabaciones:** teléfono **horizontal**, montaje
rígido en el tablero con la calle bien visible, cámara **gran angular** si
existe, 1080p, y **medir y anotar la altura de la cámara** sobre el suelo (no se
usa hoy, pero deja abierta la opción de recuperar escala por plano de tierra sin
costo).

---

## 12. CÓMO COORDINAR CON EL LADO PC

El lado PC (proyecto `Slam_GPS_Design`, Python, módulo `LMS/LMS_RL_ORB_GPS/realtime/`)
ya tiene:

- Pipeline de tiempo real con hilos y colas, **funcionando y medido**: 23 ms por
  frame, 45 ms de latencia p95, 30 Hz sostenidos
- `GpsBuffer` con acceso **causal** (nunca usa un fix que no había llegado)
- Fuentes de datos **intercambiables**: hoy leen archivos, mañana leerán la red
- Un estimador de escala por profundidad monocular que **no consume GPS**

Cuando la Fase 2 esté lista, la PC agrega una clase de ~40 líneas que lee del
socket. **No hay "fase de integración"**: el resto del sistema no se entera de
que cambió la fuente.

**Lo único que hay que acordar entre ambos lados es el protocolo (§4).** Ya está
especificado; conviene confirmarlo antes de escribir el emisor.

**Dato a reportar apenas se tenga: la frecuencia real de GPS (Fase 0.5).**
