from pyproj import Transformer
import numpy as np

def latlon_to_utm(lat, lon, alt=0.0):
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    transformer = Transformer.from_crs(f"EPSG:4326", f"EPSG:326{utm_zone}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.array([x, y, alt])