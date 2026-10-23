"""
Launcher of the project's LMS systems: one button per LMS_* folder, the
classic RL-ORB-SLAM-GPS with automatic dataset detection, and the real-time
pipeline window (realtime/live_window.py).
"""

import json
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from medir_metricas import medir_metricas

# The classic LMS_RL_ORB_GPS loads from its own folder (utils.GUI, main).
lms_rl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LMS', 'LMS_RL_ORB_GPS')
utils_gui_path = os.path.join(lms_rl_path, 'utils', 'GUI')

LMS_RL_AVAILABLE = False

if os.path.exists(utils_gui_path):
    if lms_rl_path not in sys.path:
        sys.path.insert(0, lms_rl_path)
    try:
        from utils.GUI.gui_processor import GUIProcessor
        from utils.GUI.dataset_detector import DatasetDetector
        LMS_RL_AVAILABLE = True
        print("Módulos LMS_RL_ORB_GPS cargados correctamente")
    except ImportError as e:
        print(f"Advertencia: No se pudo importar LMS_RL_ORB_GPS: {e}")
        LMS_RL_AVAILABLE = False
else:
    print(f"Advertencia: No existe el directorio {utils_gui_path}")

LMS_DIR = "LMS"


def lms_types():
    return [
        name for name in os.listdir(LMS_DIR)
        if os.path.isdir(os.path.join(LMS_DIR, name)) and name.startswith("LMS_")
    ] if os.path.exists(LMS_DIR) else []


def choose_input(lms_name):
    """A KITTI folder for the RL_ORB_GPS system, an .mp4 video for the others."""
    if "RL_ORB_GPS" in lms_name:
        folder = filedialog.askdirectory(
            title="Choose a KITTI folder (e.g. 2011_09_26_drive_0002_extract)"
        )
        if not folder:
            return None

        # image_02/data and oxts/data, in the folder chosen or one level up.
        image_dirs = []
        oxts_dirs = []
        for path in (folder, os.path.dirname(folder)):
            for root, dirs, files in os.walk(path):
                if "image_02" in dirs and "data" in os.listdir(os.path.join(root, "image_02")):
                    image_dirs.append(os.path.join(root, "image_02", "data"))
                if "oxts" in dirs and "data" in os.listdir(os.path.join(root, "oxts")):
                    oxts_dirs.append(os.path.join(root, "oxts", "data"))

        if not image_dirs or not oxts_dirs:
            messagebox.showerror("Error",
                "The folder chosen does not have the KITTI layout.\n"
                "It needs the folders image_02/data and oxts/data.\n\n"
                "Choose the folder that holds them (e.g. 2011_09_26_drive_0002_extract)")
            return None

        image_files = [f for f in os.listdir(image_dirs[0]) if f.endswith(('.png', '.jpg', '.jpeg'))]
        gps_files = [f for f in os.listdir(oxts_dirs[0]) if f.endswith('.txt')]
        if not image_files:
            messagebox.showerror("Error", "No images in image_02/data")
            return None
        if not gps_files:
            messagebox.showerror("Error", "No GPS data in oxts/data")
            return None

        # The folder that holds both.
        return os.path.dirname(os.path.dirname(image_dirs[0]))
    return filedialog.askopenfilename(
        title="Choose an .mp4 video",
        filetypes=[("Video files", "*.mp4")]
    )


def run_lms(lms_name):
    source = choose_input(lms_name)
    if not source:
        messagebox.showwarning("Cancelled", "No input was chosen.")
        return

    messagebox.showinfo("LMS chosen", f"Running: {lms_name}\nWith input:\n{source}")
    script_path = os.path.join(LMS_DIR, lms_name, "main.py")
    if os.path.isdir(source):
        command = ["python3", script_path, source, "--kitti"]
    else:
        command = ["python3", script_path, source]

    results = medir_metricas(command, lms_name)

    # Long outputs would not fit in the dialog.
    shown = {}
    for key, value in results.items():
        if isinstance(value, str):
            if not value.strip():
                continue
            if len(value) > 500:
                shown[key] = value[:500] + "... (truncated)"
                continue
        shown[key] = value

    messagebox.showinfo("Run results", json.dumps(shown, indent=4, ensure_ascii=False))


def run_lms_rl_orb_gps_auto():
    """The classic LMS_RL_ORB_GPS on a folder, detecting whether it is KITTI or a phone recording."""
    if not LMS_RL_AVAILABLE:
        messagebox.showerror(
            "Error",
            "The LMS_RL_ORB_GPS module is not available.\n"
            "Check that this folder exists:\n"
            "LMS/LMS_RL_ORB_GPS/utils/gps/"
        )
        return

    base_dir = filedialog.askdirectory(title="Choose a dataset folder (KITTI or phone recording)")
    if not base_dir:
        messagebox.showwarning("Cancelled", "No folder was chosen.")
        return

    dataset_info = DatasetDetector.detect_dataset_type(base_dir)
    summary = DatasetDetector.get_dataset_summary(dataset_info)
    if not messagebox.askyesno("Confirm dataset", f"{summary}\n\nProcess this dataset?",
                               icon='question'):
        return

    progress_window = tk.Toplevel()
    progress_window.title("Processing dataset - LMS_RL_ORB_GPS")
    progress_window.geometry("700x450")
    progress_window.resizable(True, True)

    main_frame = tk.Frame(progress_window, bg="white", padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(
        main_frame,
        text="Processing the dataset with LMS_RL_ORB_GPS",
        font=("Helvetica Neue", 14, "bold"),
        bg="white",
        fg="#0d47a1"
    ).pack(pady=(0, 10))

    text_frame = tk.Frame(main_frame, bg="white")
    text_frame.pack(fill=tk.BOTH, expand=True)
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

    # Enabled once processing ends.
    close_button = tk.Button(
        main_frame,
        text="Close",
        state=tk.DISABLED,
        font=("Segoe UI", 10, "bold"),
        bg="#388e3c",
        fg="white",
        command=progress_window.destroy
    )
    close_button.pack(pady=(10, 0))

    def append_log(message):
        progress_text.insert(tk.END, message + "\n")
        progress_text.see(tk.END)
        progress_text.update()

    def process_in_thread():
        try:
            append_log("=" * 70)
            append_log("Starting...")
            append_log("=" * 70)

            result = GUIProcessor.process_directory(
                base_path=base_dir,
                max_frames=None,
                training_mode=False,
                augmentation_prob=0.0,
                progress_callback=append_log
            )

            append_log("\n" + "=" * 70)
            if result['success']:
                append_log(" PROCESSING FINISHED")
                append_log("=" * 70)
                append_log(f"Dataset type: {result['result_type']}")
                append_log(f"Keyframes: {result.get('keyframes', 'N/A')}")
                append_log("\nResults saved in:")
                append_log("  → resultados/LMS_RL_ORB_GPS/")
                append_log("=" * 70)
                messagebox.showinfo(
                    "Processing finished",
                    f"Dataset processed\n\n"
                    f"Type: {result['result_type']}\n"
                    f"Keyframes: {result.get('keyframes', 'N/A')}\n\n"
                    f"See the folder:\n"
                    f"resultados/LMS_RL_ORB_GPS/"
                )
            else:
                append_log(" PROCESSING FAILED")
                append_log("=" * 70)
                append_log(f"Error: {result['error']}")
                if 'traceback' in result:
                    append_log("\nTraceback:")
                    append_log(result['traceback'])
                append_log("=" * 70)
                messagebox.showerror(
                    "Processing failed",
                    f"Error: {result['error']}\n\n"
                    f"The progress window has the details."
                )

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            append_log("\n" + "=" * 70)
            append_log(" UNEXPECTED ERROR")
            append_log("=" * 70)
            append_log(f"Error: {str(e)}")
            append_log("\nFull traceback:")
            append_log(error_trace)
            append_log("=" * 70)
            messagebox.showerror(
                "Unexpected error",
                f"Error while processing:\n{str(e)}\n\n"
                f"The progress window has the details."
            )

        finally:
            # From the worker thread, through after(); the window may be closed already.
            try:
                progress_window.after(0, lambda: close_button.config(state=tk.NORMAL))
                append_log("\nDone. This window can be closed.")
            except Exception as e:
                print(f"Ventana cerrada: {e}")

    threading.Thread(target=process_in_thread, daemon=True).start()


_realtime_window = None


def open_realtime_window(parent):
    """Open the real-time pipeline window, or bring it forward if it is open."""
    global _realtime_window
    if _realtime_window is not None and _realtime_window.winfo_exists():
        _realtime_window.lift()
        return
    if lms_rl_path not in sys.path:
        sys.path.insert(0, lms_rl_path)
    # Loaded on demand, so the launcher opens without loading the pipeline.
    from realtime.live_window import PipelineWindow
    _realtime_window = tk.Toplevel(parent)
    PipelineWindow(_realtime_window)


def build_launcher():
    window = tk.Tk()
    window.title("Multi-LMS platform")
    window.geometry("800x600")
    window.resizable(False, False)

    canvas = tk.Canvas(window, width=800, height=600)
    canvas.pack(fill="both", expand=True)
    # Background gradient, #99c2ff to #f0f4f8.
    for i in range(600):
        r = int(153 + (240 - 153) * (i / 600))
        g = int(194 + (244 - 194) * (i / 600))
        b = int(255 + (248 - 255) * (i / 600))
        canvas.create_line(0, i, 800, i, fill=f'#{r:02x}{g:02x}{b:02x}')

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton",
                    font=("Segoe UI", 11, "bold"),
                    padding=10,
                    background="#388e3c",
                    foreground="white",
                    relief="flat")
    style.map("TButton", background=[("active", "#2e7d32")])
    style.configure("TLabel", background="#ffffff", foreground="#333")

    frame = tk.Frame(canvas, bg="white", bd=2, relief="ridge")
    frame.place(relx=0.5, rely=0.5, anchor="center", width=600, height=470)

    tk.Label(frame, text="Multi-LMS platform", font=("Helvetica Neue", 20, "bold"), bg="white", fg="#0d47a1").pack(pady=(20, 10))
    tk.Label(frame, text="Choose an LMS and an .mp4 video to build a 2D trajectory", font=("Segoe UI", 11), bg="white").pack()

    buttons = tk.Frame(frame, bg="white")
    buttons.pack(pady=20)

    types = lms_types()
    others = [lms for lms in types if "RL_ORB_GPS" not in lms]
    has_rl_orb_gps = any("RL_ORB_GPS" in lms for lms in types)

    for i, lms in enumerate(others):
        ttk.Button(buttons, text=lms.replace("LMS_", "LMS "),
                   command=lambda lms=lms: run_lms(lms)).grid(row=i // 2, column=i % 2, padx=15, pady=10)

    # LMS_RL_ORB_GPS gets one button: with automatic detection when its modules
    # load, the classic run otherwise.
    if has_rl_orb_gps:
        row = (len(others) + 1) // 2
        if LMS_RL_AVAILABLE:
            rl_button = ttk.Button(buttons, text="LMS_RL_ORB_GPS", command=run_lms_rl_orb_gps_auto)
        else:
            lms = next(lms for lms in types if "RL_ORB_GPS" in lms)
            rl_button = ttk.Button(buttons, text="LMS_RL_ORB_GPS (classic mode)",
                                   command=lambda: run_lms(lms))
        rl_button.grid(row=row, column=0, columnspan=2, padx=15, pady=10, sticky="ew")

    # The real-time pipeline: the phone's video or a recording, without a console.
    realtime_row = (len(others) + 1) // 2 + (1 if has_rl_orb_gps else 0)
    ttk.Button(buttons, text="Real-time SLAM GPS",
               command=lambda: open_realtime_window(window)).grid(
        row=realtime_row, column=0, columnspan=2, padx=15, pady=10, sticky="ew")

    tk.Label(frame, text="© TEC 2025 | SLAM Project", font=("Segoe UI", 11), bg="white", fg="black").pack(side="bottom", pady=10)

    window.mainloop()


if __name__ == "__main__":
    build_launcher()
