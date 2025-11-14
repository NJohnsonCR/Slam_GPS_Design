import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import sys
import threading
from medir_metricas import medir_metricas

# ============================================================================
# IMPORTS PARA LMS_RL_ORB_GPS (NUEVOS)
# ============================================================================
# Agregar path del sistema LMS_RL_ORB_GPS al sys.path
lms_rl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LMS', 'LMS_RL_ORB_GPS')
utils_gui_path = os.path.join(lms_rl_path, 'utils', 'GUI')

LMS_RL_AVAILABLE = False

if os.path.exists(utils_gui_path):
    # Agregar el path al sys.path si no está ya
    if lms_rl_path not in sys.path:
        sys.path.insert(0, lms_rl_path)
    
    try:
        # Importar usando la ruta relativa correcta desde LMS_RL_ORB_GPS
        from utils.GUI.gui_processor import GUIProcessor
        from utils.GUI.dataset_detector import DatasetDetector
        LMS_RL_AVAILABLE = True
        print("Módulos LMS_RL_ORB_GPS cargados correctamente")
    except ImportError as e:
        print(f"Advertencia: No se pudo importar LMS_RL_ORB_GPS: {e}")
        LMS_RL_AVAILABLE = False
else:
    print(f"Advertencia: No existe el directorio {utils_gui_path}")
# ============================================================================

CARPETA_LMS = "LMS"

def obtener_tipos_lms():
    return [
        nombre for nombre in os.listdir(CARPETA_LMS)
        if os.path.isdir(os.path.join(CARPETA_LMS, nombre)) and nombre.startswith("LMS_")
    ] if os.path.exists(CARPETA_LMS) else []

def seleccionar_entrada(nombre_lms):
    # Determinar si este LMS específico necesita directorio KITTI en lugar de video
    necesita_kitti = "RL_ORB_GPS" in nombre_lms
    print(necesita_kitti)
    
    if necesita_kitti:
        # Seleccionar directorio KITTI
        directorio = filedialog.askdirectory(
            title="Seleccionar directorio con datos KITTI (ej: 2011_09_26_drive_0002_extract)"
        )
        if not directorio:
            return None
        
        # Verificar estructura de directorio KITTI
        # Buscar subdirectorios image_02/data y oxts/data
        image_dirs = []
        oxts_dirs = []
        
        # Buscar en el directorio seleccionado y un nivel arriba
        possible_paths = [
            directorio,
            os.path.dirname(directorio)  # Un nivel arriba
        ]
        
        for path in possible_paths:
            for root, dirs, files in os.walk(path):
                if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
                    image_dirs.append(os.path.join(root, "image_02", "data"))
                if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
                    oxts_dirs.append(os.path.join(root, "oxts", "data"))
        
        if not image_dirs or not oxts_dirs:
            messagebox.showerror("Error", 
                "El directorio seleccionado no tiene la estructura KITTI esperada.\n"
                "Se requieren las carpetas: image_02/data y oxts/data\n\n"
                "Seleccione el directorio que contiene estas carpetas (ej: 2011_09_26_drive_0002_extract)")
            return None
        
        # Verificar que hay archivos en las carpetas
        image_files = [f for f in os.listdir(image_dirs[0]) if f.endswith(('.png', '.jpg', '.jpeg'))]
        gps_files = [f for f in os.listdir(oxts_dirs[0]) if f.endswith('.txt')]
        
        if not image_files:
            messagebox.showerror("Error", "No se encontraron imágenes en image_02/data")
            return None
            
        if not gps_files:
            messagebox.showerror("Error", "No se encontraron datos GPS en oxts/data")
            return None
            
        # Devolver el directorio base que contiene ambas carpetas
        base_dir = os.path.dirname(os.path.dirname(image_dirs[0]))
        return base_dir
    else:
        # Seleccionar video como antes
        return filedialog.askopenfilename(
            title="Seleccionar video .mp4",
            filetypes=[("Archivos de video", "*.mp4")]
        )

def ejecutar_lms(nombre_lms):
    entrada = seleccionar_entrada(nombre_lms)
    if not entrada:
        messagebox.showwarning("Selección cancelada", "No se seleccionó ninguna entrada.")
        return

    messagebox.showinfo("LMS seleccionado", f"Ejecutando: {nombre_lms}\nCon entrada:\n{entrada}")
    script_path = os.path.join(CARPETA_LMS, nombre_lms, "main.py")
    
    # Determinar el tipo de entrada (directorio KITTI o archivo video)
    if os.path.isdir(entrada):
        comando = ["python3", script_path, entrada, "--kitti"]
    else:
        comando = ["python3", script_path, entrada]

    resultados = medir_metricas(comando, nombre_lms)

    salida_filtrada = {}
    for clave, valor in resultados.items():
        if isinstance(valor, str):
            if not valor.strip():
                continue
            if len(valor) > 500:
                salida_filtrada[clave] = valor[:500] + "... (truncado)"
                continue
        salida_filtrada[clave] = valor

    mensaje = json.dumps(salida_filtrada, indent=4, ensure_ascii=False)
    messagebox.showinfo("Resultados de ejecución", mensaje)

# ============================================================================
# NUEVA FUNCIÓN PARA LMS_RL_ORB_GPS CON DETECCIÓN AUTOMÁTICA
# ============================================================================
def ejecutar_lms_rl_orb_gps_auto():
    """
    Ejecuta el sistema LMS_RL_ORB_GPS con detección automática de dataset.
    NO modifica el flujo existente de otros LMS.
    """
    if not LMS_RL_AVAILABLE:
        messagebox.showerror(
            "Error",
            "El módulo LMS_RL_ORB_GPS no está disponible.\n"
            "Verifique que exista la carpeta:\n"
            "LMS/LMS_RL_ORB_GPS/utils/gps/"
        )
        return
    
    # Seleccionar directorio base
    directorio_base = filedialog.askdirectory(
        title="Seleccionar Directorio Base (KITTI o Mobile Dataset)"
    )
    
    if not directorio_base:
        messagebox.showwarning("Selección cancelada", "No se seleccionó ningún directorio.")
        return
    
    # Detectar rápidamente el tipo para mostrar preview
    dataset_info = DatasetDetector.detect_dataset_type(directorio_base)
    
    # Mostrar ventana de confirmación con información del dataset
    summary = DatasetDetector.get_dataset_summary(dataset_info)
    
    confirm = messagebox.askyesno(
        "Confirmar Dataset",
        f"{summary}\n\n¿Desea procesar este dataset?",
        icon='question'
    )
    
    if not confirm:
        return
    
    # Crear ventana de progreso
    progress_window = tk.Toplevel()
    progress_window.title("Procesando Dataset - LMS_RL_ORB_GPS")
    progress_window.geometry("700x450")
    progress_window.resizable(True, True)
    
    # Frame principal con padding
    main_frame = tk.Frame(progress_window, bg="white", padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Título
    title_label = tk.Label(
        main_frame,
        text="Procesando Dataset con LMS_RL_ORB_GPS",
        font=("Helvetica Neue", 14, "bold"),
        bg="white",
        fg="#0d47a1"
    )
    title_label.pack(pady=(0, 10))
    
    # Frame para el texto con scrollbar
    text_frame = tk.Frame(main_frame, bg="white")
    text_frame.pack(fill=tk.BOTH, expand=True)
    
    # Text widget con scrollbar
    progress_text = tk.Text(
        text_frame,
        wrap=tk.WORD,
        font=("Consolas", 10),
        bg="#f5f5f5",
        fg="#333333",
        relief="solid",
        borderwidth=1
    )
    progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(text_frame, command=progress_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    progress_text.config(yscrollcommand=scrollbar.set)
    
    # Botón de cerrar (inicialmente deshabilitado)
    close_button = tk.Button(
        main_frame,
        text="Cerrar",
        state=tk.DISABLED,
        font=("Segoe UI", 10, "bold"),
        bg="#388e3c",
        fg="white",
        command=progress_window.destroy
    )
    close_button.pack(pady=(10, 0))
    
    def append_log(message):
        """Agregar mensaje al log de progreso."""
        progress_text.insert(tk.END, message + "\n")
        progress_text.see(tk.END)
        progress_text.update()
    
    def process_in_thread():
        """Ejecutar procesamiento en hilo separado."""
        try:
            append_log("=" * 70)
            append_log("Iniciando procesamiento...")
            append_log("=" * 70)
            
            result = GUIProcessor.process_directory(
                base_path=directorio_base,
                max_frames=None,  # Procesar todos los frames
                training_mode=False,
                augmentation_prob=0.0,
                progress_callback=append_log
            )
            
            append_log("\n" + "=" * 70)
            
            if result['success']:
                append_log(" PROCESAMIENTO COMPLETADO EXITOSAMENTE")
                append_log("=" * 70)
                append_log(f"Tipo de dataset: {result['result_type']}")
                append_log(f"Keyframes generados: {result.get('keyframes', 'N/A')}")
                append_log("\nResultados guardados en:")
                append_log("  → resultados/LMS_RL_ORB_GPS/")
                append_log("=" * 70)
                
                messagebox.showinfo(
                    "Procesamiento Completo",
                    f"Dataset procesado exitosamente\n\n"
                    f"Tipo: {result['result_type']}\n"
                    f"Keyframes: {result.get('keyframes', 'N/A')}\n\n"
                    f"Revisa la carpeta:\n"
                    f"resultados/LMS_RL_ORB_GPS/"
                )
            else:
                append_log(" ERROR EN PROCESAMIENTO")
                append_log("=" * 70)
                append_log(f"Error: {result['error']}")
                if 'traceback' in result:
                    append_log("\nTraceback:")
                    append_log(result['traceback'])
                append_log("=" * 70)
                
                messagebox.showerror(
                    "Error en Procesamiento",
                    f"Error: {result['error']}\n\n"
                    f"Revisa la ventana de progreso para más detalles."
                )
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            
            append_log("\n" + "=" * 70)
            append_log(" ERROR INESPERADO")
            append_log("=" * 70)
            append_log(f"Error: {str(e)}")
            append_log("\nTraceback completo:")
            append_log(error_trace)
            append_log("=" * 70)
            
            messagebox.showerror(
                "Error Inesperado",
                f"Error durante procesamiento:\n{str(e)}\n\n"
                f"Revisa la ventana de progreso para más detalles."
            )
        
        finally:
            # Habilitar botón de cerrar cuando termine (usar after para thread safety)
            try:
                progress_window.after(0, lambda: close_button.config(state=tk.NORMAL))
                append_log("\nProcesamiento finalizado. Puede cerrar esta ventana.")
            except Exception as e:
                # La ventana pudo haber sido cerrada manualmente
                print(f"Ventana cerrada: {e}")
    
    # Iniciar procesamiento en hilo separado
    thread = threading.Thread(target=process_in_thread, daemon=True)
    thread.start()
# ============================================================================

def crear_interfaz():
    ventana = tk.Tk()
    ventana.title("Plataforma LMS múltiple")
    ventana.geometry("800x500")
    ventana.resizable(False, False)

    # === Canvas con degradado ===
    canvas = tk.Canvas(ventana, width=800, height=500)
    canvas.pack(fill="both", expand=True)

    # Degradado de #99c2ff a #f0f4f8
    for i in range(500):
        r = int(153 + (240 - 153) * (i / 500))  # Red: 153 → 240
        g = int(194 + (244 - 194) * (i / 500))  # Green: 194 → 244
        b = int(255 + (248 - 255) * (i / 500))  # Blue: 255 → 248
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, 800, i, fill=color)

    estilo = ttk.Style()
    estilo.theme_use('clam')
    estilo.configure("TButton",
                     font=("Segoe UI", 11, "bold"),
                     padding=10,
                     background="#388e3c",
                     foreground="white",
                     relief="flat")
    estilo.map("TButton", background=[("active", "#2e7d32")])
    estilo.configure("TLabel", background="#ffffff", foreground="#333")

    frame = tk.Frame(canvas, bg="white", bd=2, relief="ridge")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=350)

    tk.Label(frame, text="Plataforma LMS múltiple", font=("Helvetica Neue", 20, "bold"), bg="white", fg="#0d47a1").pack(pady=(20, 10))
    tk.Label(frame, text="Seleccione un LMS y un video .mp4 para generar una trayectoria 2D", font=("Segoe UI", 11), bg="white").pack()

    # === Botones ===
    frame_botones = tk.Frame(frame, bg="white")
    frame_botones.pack(pady=20)

    tipos_lms = obtener_tipos_lms()
    
    # ========================================================================
    # MODIFICACIÓN: Usar detección automática para LMS_RL_ORB_GPS
    # ========================================================================
    lms_normales = [lms for lms in tipos_lms if "RL_ORB_GPS" not in lms]
    tiene_rl_orb_gps = any("RL_ORB_GPS" in lms for lms in tipos_lms)
    
    # Botones para LMS normales (flujo existente sin cambios)
    for i, lms in enumerate(lms_normales):
        nombre_mostrado = lms.replace("LMS_", "LMS tipo ")
        boton = ttk.Button(frame_botones, text=nombre_mostrado, command=lambda lms=lms: ejecutar_lms(lms))
        boton.grid(row=i // 2, column=i % 2, padx=15, pady=10)
    
    # Botón único para LMS_RL_ORB_GPS (con auto-detección integrada)
    if tiene_rl_orb_gps:
        row_especial = (len(lms_normales) + 1) // 2
        
        if LMS_RL_AVAILABLE:
            # Si los módulos están disponibles, usar detección automática
            boton_rl_orb = ttk.Button(
                frame_botones,
                text="LMS_RL_ORB_GPS",
                command=ejecutar_lms_rl_orb_gps_auto
            )
        else:
            # Si no están disponibles, usar el modo clásico como fallback
            for lms in tipos_lms:
                if "RL_ORB_GPS" in lms:
                    boton_rl_orb = ttk.Button(
                        frame_botones,
                        text="LMS_RL_ORB_GPS (Modo Clásico)",
                        command=lambda lms=lms: ejecutar_lms(lms)
                    )
                    break
        
        boton_rl_orb.grid(row=row_especial, column=0, columnspan=2, padx=15, pady=10, sticky="ew")
    # ========================================================================

    tk.Label(frame, text="© TEC 2025 | Proyecto SLAM", font=("Segoe UI", 11), bg="white", fg="black").pack(side="bottom", pady=10)

    ventana.mainloop()

if __name__ == "__main__":
    crear_interfaz()