import os
import simplekml

def kitti_oxts_to_kml(oxts_folder, output_file="trajectory.kml"):
    """
    Lee los archivos OXTS de KITTI y genera un archivo KML para Google Earth.
    
    Args:
        oxts_folder (str): Ruta a la carpeta `oxts/data` de KITTI.
        output_file (str): Nombre del archivo de salida .kml
    """
    kml = simplekml.Kml()
    coords = []

    # Iterar archivos en orden (KITTI los nombra 0000000000.txt, 0000000001.txt, ...)
    files = sorted(os.listdir(oxts_folder))

    for f in files:
        if f.endswith(".txt"):
            path = os.path.join(oxts_folder, f)
            with open(path, "r") as fh:
                line = fh.readline().strip().split()
                if len(line) < 2:
                    continue
                lat = float(line[0])
                lon = float(line[1])
                coords.append((lon, lat))  # Google Earth usa (lon, lat)

    # Crear línea en KML
    if coords:
        linestring = kml.newlinestring(name="KITTI Trajectory", coords=coords)
        linestring.style.linestyle.width = 3
        linestring.style.linestyle.color = simplekml.Color.red

        # Guardar archivo
        kml.save(output_file)
        print(f"✅ Archivo KML guardado en: {output_file}")
    else:
        print("⚠️ No se encontraron coordenadas válidas.")

# Ejemplo de uso:
if __name__ == "__main__":
    # Cambia esta ruta a donde tengas KITTI descargado
    oxts_folder = "kitti_data/2011_09_26/2011_09_26_drive_0009_sync/oxts/data"
    kitti_oxts_to_kml(oxts_folder, "kitti_trajectory.kml")
