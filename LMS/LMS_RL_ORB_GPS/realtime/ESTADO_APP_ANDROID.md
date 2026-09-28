# Estado de la app Android

Traspaso del task de la app móvil al task del pipeline. Resume lo implementado, las decisiones tomadas y lo que el lado PC necesita saber para conectarse. Actualizado al 27/9/2026.

`CONTEXTO_APP_ANDROID.md` sigue siendo el plan original. Este documento describe lo que efectivamente se construyó y en qué se aparta de ese plan.

- Repositorio de la app: `~/Documents/GitHub/Slam_GPS_Design_App`, rama `develop`.
- El protocolo completo también está en el `README.md` de ese repositorio, sección "Transmisión en vivo".

---

## 1. Reglas entre los dos tasks

- **El repositorio de la app es de solo lectura para el task del pipeline.** Cualquier cambio en el teléfono (protocolo, formatos, comportamiento) se pide al task de la app.
  La configuración local de este proyecto (`.claude/settings.local.json`) lo refuerza: bloquea la edición de archivos del repositorio de la app.
- **Lo único que une a los dos lados es el protocolo de red** (sección 4). El pipeline implementa su propio cliente; no importa ni ejecuta código de la app.
- Los scripts de `tools/` de la app sirven como ejemplo de referencia de un cliente que funciona.

---

## 2. Estado

Las fases 0 a 4 del plan están implementadas y probadas en el teléfono, salvo la prueba de campo.

| Fase | Estado |
|---|---|
| 0. Compilar y correr | Hecha |
| 0.5. Medir la frecuencia de GPS | Hecha: 1.91 Hz |
| 1. GPS a archivo | Hecha |
| 2. Sensores por UDP | Hecha (solo GPS; la IMU no se transmite) |
| 3. Video por TCP | Hecha |
| 4. Control y modos | Hecha, salvo la prueba de campo en vehículo |

Los requisitos imprescindibles del §7 del plan están cumplidos, incluidos la pantalla de ajustes (puertos y calidad JPEG) y la frecuencia de GPS en la pantalla del teléfono.

No se hicieron los "deseables": IMU por la red y selección de la cámara gran angular.

---

## 3. Decisiones tomadas

1. **El control es desde la PC**, con el botón Record del teléfono como respaldo.
2. **Dos modos de uso**, elegidos en la interfaz de la PC:
   - *Grabar y transmitir* (el ideal): la PC recibe en vivo y el teléfono guarda una copia completa.
   - *Solo transmitir*: la PC recibe en vivo y el teléfono no ocupa espacio.
3. **El teléfono transmite siempre que haya una PC conectada**, grabe o no. `START` y `STOP` controlan solo la grabación. Así, antes de arrancar, la PC ya ve el video y el GPS y puede verificar el encuadre y la señal.
4. **Guardar la trayectoria del recorrido es trabajo del lado PC.** La respuesta a `START` incluye el nombre de la carpeta de la grabación en el teléfono, para guardarlo junto con la trayectoria y compararlas después.
5. **El teléfono va montado en horizontal, con la cámara a la izquierda.** En esa posición la imagen transmitida y la grabada quedan derechas.
6. **`gyro_accel.csv` sigue el formato del §6 del plan**, con 8 columnas y sin magnetómetro.
7. **`movie.mp4` lleva una marca de rotación** para que OpenCV lo entregue en horizontal, igual que la transmisión.
8. **La integración con el pipeline va antes de la prueba de campo.** Primero se une la app con el pipeline en el escritorio, después se aplica la fusión de datos, y la prueba de campo se hace con todo junto.

---

## 4. Protocolo

El teléfono es el servidor y la PC se conecta a él. La pantalla de video del teléfono muestra su IP. Teléfono y PC deben estar en la misma red; en campo, con el S23 como punto de acceso.

Los puertos por defecto son 5000 (video) y 5001 (GPS y control). Se pueden cambiar en los ajustes de la app, igual que la calidad del JPEG (80 por defecto). Si se cambian, el pipeline debe usar los mismos valores.

### 4.1 Video: TCP, puerto 5000

Al conectarse, la PC recibe una línea de texto con los parámetros de la cámara, en píxeles de la imagen transmitida:

```
HELLO,<ancho>,<alto>,<fx>,<fy>,<cx>,<cy>,<fps>\n
```

Valores reales del Galaxy S23:

```
HELLO,1280,720,867.81,868.55,630.75,367.79,30
```

Después, por cada frame, un encabezado de 20 bytes en big-endian, seguido del JPEG:

| Campo | Bytes | Tipo |
|---|---|---|
| Longitud del JPEG | 4 | uint32 |
| Instante de captura, reloj monótono del teléfono (ns) | 8 | uint64 |
| Instante de captura, hora Unix (ns) | 8 | uint64 |

- Los frames no se encolan. Si la red está ocupada, el teléfono descarta frames, de modo que el retraso no crece.
- El teléfono atiende a una PC a la vez; una conexión nueva reemplaza a la anterior. Para reconectarse basta con volver a conectarse.
- Por WiFi llegan menos de 30 fps. El pipeline debe guiarse por las marcas de tiempo, no por el número de frame.

### 4.2 GPS, estado y control: UDP, puerto 5001

Mensajes de la PC al teléfono:

| Mensaje | Efecto |
|---|---|
| `SUBSCRIBE` | Registra a la PC como destino de los datos. Hay que repetirlo cada ~2 s; si pasan 5 s sin recibirlo, el teléfono muestra a la PC como inactiva. |
| `START` | Inicia la grabación, igual que el botón Record. |
| `STOP` | Detiene la grabación. |

Mensajes del teléfono a la PC:

| Mensaje | Cuándo |
|---|---|
| `SUBSCRIBED` | En respuesta a cada `SUBSCRIBE`. |
| `STATE,RECORDING,<carpeta>` o `STATE,IDLE` | En respuesta a cada `SUBSCRIBE`, `START` y `STOP`, y cuando la grabación se inicia o detiene con el botón del teléfono. `IDLE` significa que no graba; la transmisión sigue. |
| `GPS,<t_ns>,<lat>,<lon>,<alt>,<speed>,<unix_ns>` | Cada posición, grabe o no. Es la misma fila de `location.csv`. |

- UDP puede perder mensajes. La PC debe repetir `START` o `STOP` hasta recibir el estado esperado; repetirlos no tiene efecto si el teléfono ya está en ese estado.
- La PC debe usar **el mismo socket UDP** para `SUBSCRIBE` y para las órdenes.
- Si la app no está en primer plano en la pantalla de video, las órdenes se ignoran y el teléfono responde con su estado actual.
- En las pruebas se perdió 1 posición de 60 en el WiFi de la casa. El pipeline debe tolerar posiciones faltantes.

### 4.3 Clientes de ejemplo

En el repositorio de la app, carpeta `tools/`:

| Script | Qué hace |
|---|---|
| `video_receiver.py` | Recibe y muestra el video; informa fps, ancho de banda y retraso. Necesita OpenCV. |
| `udp_gps_receiver.py` | Se suscribe y muestra las posiciones y los cambios de estado. |
| `phone_control.py` | Envía `start` o `stop` y reintenta hasta la confirmación. |

---

## 5. Datos y formatos

### 5.1 Relojes

- **Todas las marcas de tiempo principales usan el mismo reloj**, `SystemClock.elapsedRealtimeNanos()`. Esto vale para la primera columna de cada archivo, la marca de cada frame transmitido y el `<t_ns>` del GPS. El Galaxy S23 reporta `SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME`.
- **Hay que alinear por ese reloj, no por la hora Unix.** En los archivos de cámara (`movie_metadata.csv` y `frame_timestamps.txt`), la columna Unix es la hora en que el frame llegó a la app, entre 60 y 75 ms después de la captura. En la IMU y el GPS no hay ese retraso.
- En la transmisión de video, en cambio, la hora Unix es la del instante de captura.

### 5.2 Archivos de cada grabación

Todos coinciden con el §6 del plan:

| Archivo | Formato |
|---|---|
| `location.csv` | `Timestamp[nanosecond],latitude[degrees],longitude[degrees],altitude[meters],speed[meters/second],Unix time[nanosecond]` |
| `movie_metadata.csv` | 12 columnas, con `fx`, `fy` ya en píxeles de la imagen de 1280x720 (antes MARS Logger escribía 2766, en píxeles del sensor). |
| `gyro_accel.csv` | `Timestamp[nanosec],gx,gy,gz,ax,ay,az,Unix time[nanosec]` |
| `frame_timestamps.txt` | `Frame timestamp[nanosec],Unix time[nanosec]`, con las marcas redondeadas al microsegundo. |

### 5.3 Video grabado y video transmitido

- Mientras se graba, **cada frame transmitido es el mismo que se guarda en `movie.mp4`, con la misma marca de tiempo**. Se verificó con 1524 de 1524 frames emparejados; la diferencia de imagen fue de 0.2 px, solo compresión.
- Leído con OpenCV, `movie.mp4` sale en 1280x720, en la misma orientación que la transmisión. Los píxeles están guardados en vertical con una marca de rotación de 270°, que OpenCV aplica por defecto.
- Con esa orientación, los intrínsecos del `HELLO` valen también para el video grabado.
- **Los primeros ~6 frames de cada `movie.mp4` (0.2 s) no muestran la imagen de su marca de tiempo**: el codificador tarda en ponerse al día. El pipeline debe descartarlos si usa el video grabado.
- **Los frames no siempre están separados exactamente por 33.3 ms.** Cuando la exposición automática cambia, el instante de captura se corre (21 de 21 irregularidades coincidieron con cambios de exposición).
- Las grabaciones anteriores (`mobile_data/2025_03_11/`) son de otra app y están en vertical (720x1280).

### 5.4 Cámara

- **Intrínsecos:** los del `HELLO`. Salen de la calibración de fábrica (`LENS_INTRINSIC_CALIBRATION`), escalada a 1280x720. El centro óptico está ~9 px a la izquierda y ~8 px abajo del centro de la imagen.
- **Distorsión:** el teléfono **no** la corrige, y el `HELLO` no la incluye. En las esquinas los puntos se desplazan hasta ~17 px.
  - En el orden de Android (k1, k2, k3, p1, p2): `[0.0903, -0.1210, 0.0544, 0, 0]`.
  - En el orden de OpenCV (k1, k2, p1, p2, k3): `[0.0903, -0.1210, 0, 0, 0.0544]`.
- **Obturador rodante:** 6.54 ms de lectura por frame.
- **Cámara usada:** la principal, de 5.40 mm. La cámara "0" del S23 es lógica; la app usa la física "5".

### 5.5 Otros datos del Galaxy S23

| Dato | Valor |
|---|---|
| GPS | 1.91 Hz caminando al aire libre, ~1.2 Hz bajo techo. No es 1 Hz como en las grabaciones viejas. |
| Velocidad del GPS | Doppler: mucho menos ruidosa que derivar las posiciones. |
| IMU | ~53 Hz, solo en archivo. |
| Video | 30 fps. |

---

## 6. Mediciones de la conexión

| Medición | WiFi de la casa | Cable USB |
|---|---|---|
| Frames por segundo recibidos | ~27 de promedio, bajones a 15 | 30.0 constantes |
| Frames perdidos | varios por segundo en los bajones | 0 |
| Tamaño por frame (calidad 80) | 15–260 KB según la escena, ~130 KB lo típico | igual |
| Tamaño por frame (calidad 60) | ~65 KB lo típico, la mitad que con 80 | igual |
| Ancho de banda | 0.5–7 MB/s, ~4 MB/s lo típico | igual |
| Retraso de captura a llegada | ~76 ms | ~76 ms |

- El teléfono comprime cada JPEG en 5–9 ms, así que no es el cuello de botella. Los bajones son de la red. Falta medir con el S23 como hotspot.
- Casi todo el retraso lo pone la propia cámara (60–75 ms). La transmisión agrega 10–20 ms.
- Con la transmisión activa, la grabación no perdió ningún frame, ni siquiera cuando la PC dejó de leer.

---

## 7. Observaciones para el pipeline

El task de la app leyó el código de `realtime/` solo en parte (sobre todo `sources.py`). Estas observaciones hay que verificarlas en el task del pipeline.

1. **`load_mobile_session` (`sources.py`) alinea por la columna Unix**, suponiendo que la app la escribe con un solo reloj. No es así para los archivos de cámara (5.1). Hay que alinear por la primera columna.
2. **Usar los intrínsecos del `HELLO`** en lugar de valores escritos en el código.
3. **Corregir o considerar la distorsión** del lente (5.4).
4. **Guiarse por la marca de tiempo de cada frame y de cada posición**, y tolerar frames y posiciones faltantes.
5. **La frecuencia real de GPS es ~1.9 Hz.** Los supuestos basados en 1 Hz pueden revisarse.
6. **La velocidad Doppler del GPS** podría mejorar la línea base de escala por GPS (§3.3 y §3.5.1 de `scripts/rl/CONTEXTO.md`). Es una idea a evaluar, no una conclusión.
7. **En el entorno virtual del proyecto están instalados `opencv-python` y `opencv-contrib-python` a la vez.** Ambos instalan `cv2` y pueden chocar. Conviene dejar uno.
8. **En OpenCV, los nombres de ventana deben ir sin tildes.** El backend Qt no encuentra ventanas con nombres no ASCII, y `imshow` abre una ventana nueva por frame.

---

## 8. Cómo arrancar una sesión

### Preparación

- **Teléfono y PC en la misma red.**
  - En el escritorio: el teléfono en el WiFi de la casa (hoy 192.168.100.108) y la PC por cable en la misma red (192.168.100.18).
  - En campo: el S23 como punto de acceso.
- **Ajustes del teléfono**, sección *Streaming to the PC*: 5000, 5001 y calidad 80, salvo que se decida otra cosa.
- Para diagnosticar desde la PC:
  - `adb` está en `~/Android/Sdk/platform-tools/adb` (no está en el PATH).
  - OpenCV está en el entorno virtual de este proyecto.

### Cada sesión

1. Abrir la app en la pantalla de video. La línea de estado muestra la IP del teléfono y si hay señal de GPS.
2. La PC se conecta por TCP al puerto 5000 y lee el `HELLO`.
3. La PC abre un socket UDP, envía `SUBSCRIBE` al puerto 5001 y lo repite cada ~2 s durante toda la sesión. Por ese mismo socket recibe `SUBSCRIBED`, `STATE` y las líneas `GPS`.
4. Según el modo:
   - *Grabar y transmitir:* la PC envía `START` hasta recibir `STATE,RECORDING,<carpeta>` y guarda el nombre de la carpeta. Al terminar, envía `STOP` hasta recibir `STATE,IDLE`.
   - *Solo transmitir:* la PC no envía órdenes.
5. Si la conexión de video se corta o deja de llegar datos por unos segundos, la PC vuelve a conectarse y recibe un `HELLO` nuevo. Si deja de llegar `SUBSCRIBED`, el teléfono no es alcanzable.

### Verificación sin el pipeline

Los scripts de `tools/` de la app prueban la conexión por separado:

- Si `video_receiver.py` y `udp_gps_receiver.py` reciben bien, la app y la red funcionan, y cualquier falla está en el pipeline.
- Si ellos también fallan, el problema es de la app o de la red y va al task de la app.

---

## 9. Prueba de campo

Se hace al final, con el pipeline integrado. Qué revisar:

- **Frames por segundo y frames perdidos con el hotspot.** Si el WiFi no alcanza, bajar la calidad JPEG a 60 en los ajustes; cada frame pesa la mitad.
- **Frecuencia del GPS en el vehículo.** La pantalla del teléfono la muestra y avisa si pasan más de 3 s sin posición.
- **Que la grabación de respaldo quede completa**, en el modo grabar y transmitir.

---

## 10. Cuándo volver al task de la app

- **Si la fusión necesita la IMU en vivo.** Hoy la IMU solo se graba en archivo, a ~53 Hz. Transmitirla seguiría el mismo esquema que el GPS, con líneas `IMU,...` como proponía el §4.3 del plan.
- **Ante cualquier cambio** de protocolo, formato o comportamiento del teléfono.
- **Ante problemas que se reproduzcan con los scripts de `tools/`**, sin el pipeline.
