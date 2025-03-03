from __future__ import annotations
# froggeolib
from geographiclib.geodesic import Geodesic
from dataclasses import dataclass, asdict
import mgrs
import math
import geojson
import json
import struct

# monkey patch since this is fucked
_original_default = json.JSONEncoder().default

def _patched_default(self, obj):
    if hasattr(obj, 'json') and callable(obj.json):
        return obj.json()
    return _original_default(obj)

json.JSONEncoder.default = _patched_default

class PosObject(): #unused for now
    def __init__(self, lat:float, lon:float, alt):
        self.lat = lat
        self.lon = lon
        self.alt = float(alt)


@dataclass
class GPSposition:
    """A class representing a GPS position with latitude, longitude, altitude, and optional errors."""
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    ce: float = 0.0  # Circular error
    le: float = 0.0  # Linear error

    @classmethod
    def from_json(cls, json_dict: dict):
        """Create a GPSposition instance from a JSON dictionary."""
        return cls(
            lat=float(json_dict.get("lat", 0.0)),
            lon=float(json_dict.get("lon", 0.0)),
            alt=float(json_dict.get("alt", 0.0)),
            ce=float(json_dict.get("ce", 0.0)),
            le=float(json_dict.get("le", 0.0))
        )

    @classmethod
    def from_tuple(cls, tup: tuple):
        """Create a GPSposition instance from a tuple."""
        if len(tup) < 2:
            raise ValueError("Tuple must have at least two elements (lat, lon)")
        lat, lon = map(float, tup[:2])
        alt = float(tup[2]) if len(tup) > 2 else 0.0
        ce = float(tup[3]) if len(tup) > 3 else 0.0
        le = float(tup[4]) if len(tup) > 4 else 0.0
        return cls(lat, lon, alt, ce, le)

    def __str__(self):
        """Return a string representation of the GPS position."""
        base = f"Latitude: {self.lat:.8f} Longitude: {self.lon:.8f} Altitude: {self.alt:.3f}"
        if self.ce != 0 or self.le != 0:  # Include errors if either is non-zero
            return f"{base} CE: {self.ce:.1f} LE: {self.le:.1f}"
        return base

    def latlon(self):
        """Return latitude and longitude as a tuple."""
        return (self.lat, self.lon)

    def mgrs(self):
        """Convert the position to MGRS format using the mgrs library."""
        milobj = mgrs.MGRS()
        return milobj.toMGRS(self.lat, self.lon)

    def json(self):
        """Return a dictionary representation for JSON serialization."""
        return asdict(self)

class PosVector():
    def __init__(self, distance, azimuth, elevation):
        self.dist = distance
        self.az = azimuth
        self.elev = elevation
    def __str__(self):
        s = "Distance: {:.3f} Azimuth: {:.3f} Elevation: {:.3f}".format(self.dist, self.az, self.elev)
        return s
    def json(self):
        return json.dumps({
            "dist": self.dist,
            "az": self.az,
            "elev": self.elev
        })

class InavWaypoint():
    def __init__(self, wp_no:int, action:int, lat:float, lon:float, alt:int, p1:int, p2:int, p3:int, flag:int):
        self.pos = GPSposition(lat, lon, alt)
        self.wp_no = int(wp_no)
        self.action = int(action)
        self.p1 = int(p1)
        self.p2 = int(p2)
        self.p3 = int(p3)
        self.flag = int(flag)
    def __str__(self):
        s = f"WP No.: {self.wp_no} {self.pos} Action: {self.action} P1: {self.p1} P2: {self.p2} P3: {self.p3} Flag: {self.flag}"
        return s
    def packed(self):
        msp_wp = struct.pack('<BBiiihhhB', self.wp_no, self.action, int(self.pos.lat * 1e7), int(self.pos.lon * 1e7), altitude*100, p1, p2, p3, flag)
        return msp_wp

def convert_geopaste(string): # from gnome-maps
    x = string.split(';')[0].split(':')[1].split(',')
    return GPSposition(float(x[0]),float(x[1]),float(0))

from froggeolib import mgrs

def latlon_to_mgrs(lat: float, lon: float, *, degrees: bool = True, precision: int = 5) -> str:
    """
    Convert latitude and longitude to an MGRS string.

    Parameters:
        lat (float): Latitude in decimal degrees (if degrees=True) or radians (if degrees=False).
        lon (float): Longitude in decimal degrees (if degrees=True) or radians (if degrees=False).
        degrees (bool): Indicates if the provided lat/lon are in decimal degrees.
                        If False, they are assumed to be in radians. Default is True.
        precision (int): The MGRS precision level (number of digit pairs for easting/northing).
                         Typical values:
                           - 1 for 10 km grid squares,
                           - 2 for 1 km,
                           - 3 for 100 m,
                           - 4 for 10 m,
                           - 5 for 1 m.
                         Default is 5 (1 m precision).

    Returns:
        str: The MGRS string representation.
    """
    mgrs_obj = mgrs.MGRS()
    return mgrs_obj.toMGRS(lat, lon, inDegrees=degrees, MGRSPrecision=precision)


def mgrs_to_latlon(mgrs_str: str, *, degrees: bool = True) -> tuple:
    """
    Convert an MGRS string to latitude and longitude.

    Parameters:
        mgrs_str (str): The MGRS coordinate string.
        degrees (bool): If True, returns lat/lon in decimal degrees; if False, returns in radians.
                        Default is True.

    Returns:
        tuple: (latitude, longitude) in the specified units.
    """
    mgrs_obj = mgrs.MGRS()
    return mgrs_obj.toLatLon(mgrs_str, inDegrees=degrees)


def mgrs_to_utm(mgrs_str: str, *, encoding: str = "utf-8") -> tuple:
    """
    Convert an MGRS string to UTM coordinates.

    Parameters:
        mgrs_str (str): The MGRS coordinate string.
        encoding (str): The character encoding for the input string. Default is "utf-8".

    Returns:
        tuple: (zone, hemisphere, easting, northing)
    """
    mgrs_obj = mgrs.MGRS()
    return mgrs_obj.MGRSToUTM(mgrs_str, encoding=encoding)


def utm_to_mgrs(zone: int, hemisphere: str, easting: float, northing: float, *, precision: int = 5) -> str:
    """
    Convert UTM coordinates to an MGRS string.

    Parameters:
        zone (int): The UTM zone number (typically 1 through 60).
        hemisphere (str): 'N' for Northern Hemisphere or 'S' for Southern Hemisphere.
        easting (float): The UTM easting value.
        northing (float): The UTM northing value.
        precision (int): The MGRS precision level (see latlon_to_mgrs for details).
                         Default is 5 (1 m precision).

    Returns:
        str: The MGRS coordinate string.
    """
    mgrs_obj = mgrs.MGRS()
    return mgrs_obj.UTMToMGRS(zone, hemisphere, easting, northing, MGRSPrecision=precision)


def gps_to_vector(latlon1, latlon2):
    geod = Geodesic.WGS84
    g = geod.Inverse(latlon1.lat, latlon1.lon, latlon2.lat, latlon2.lon)
    az = g['azi1']
    dist = g['s12']
    if az<0:
        az = az+360
    if latlon1.alt > latlon2.alt:
        relalt = latlon1.alt - latlon2.alt
        elev = math.degrees( math.atan( relalt / dist ) ) * -1
    else:
        relalt = latlon2.alt - latlon1.alt
        elev = math.degrees( math.atan( relalt / dist ) ) 

    return PosVector(dist, az, elev) #dist, azimuth, elev

def vector_to_gps(latlon, dist, az):
    geod = Geodesic.WGS84
    g = geod.Direct(latlon.lat, latlon.lon, az, dist)
    return GPSposition(float(g['lat2']),float(g['lon2']),float(0))

def vector_to_gps_air(latlon, az, ang): #only valid if both points are at same ground level
    geod = Geodesic.WGS84
    truerange = math.tan(math.radians(ang)) * latlon.alt
    slantrange = latlon.alt / math.cos(math.radians(ang))
    g = geod.Direct(latlon.lat, latlon.lon, az, truerange)
    return GPSposition(float(g['lat2']),float(g['lon2']),float(0))

def vector_rangefinder_to_gps_air(latlon, az, ang, slantrange):
    geod = Geodesic.WGS84
    truerange = math.cos(math.radians(ang))*slantrange
    g = geod.Direct(latlon.lat, latlon.lon, az, truerange)
    return GPSposition(float(g['lat2']),float(g['lon2']),float(0))


def distance_m(p1: GPSposition, p2: GPSposition) -> float:
    inv = geod.Inverse(p1.lat, p1.lon, p2.lat, p2.lon)
    return inv["s12"]

def to_local_xy(origin: GPSposition, point: GPSposition):
    """
    Projects 'point' into a local tangent plane with 'origin' as (0, 0).
    x-axis points East, y-axis points North (approx).
    """
    inv = geod.Inverse(origin.lat, origin.lon, point.lat, point.lon)
    dist = inv["s12"]
    az   = inv["azi1"]  # azimuth from origin to point, relative to north
    azr  = math.radians(az)
    x = dist * math.sin(azr)  # East
    y = dist * math.cos(azr)  # North
    return x, y

def point_in_polygon(point: GPSposition, polygon: list[GPSposition]) -> bool:
    """
    Ray casting in a local 2D plane around the first polygon vertex.
    """
    # Project all polygon vertices + the point to local XY
    origin = polygon[0]
    poly_xy = [to_local_xy(origin, v) for v in polygon]
    px, py  = to_local_xy(origin, point)

    # Standard ray-casting count
    inside = False
    for i in range(len(poly_xy)):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
        cond = ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / (y2 - y1) + x1
        )
        if cond:
            inside = not inside
    return inside

def point_in_shape(pos: GPSposition, shape_def: dict) -> bool:
    """
    shape_def examples:
      {"shape": "circle",  "points": [center],             "size": 100}
      {"shape": "polygon", "points": [p1,p2,..., pN],      "size": None}
    """
    shape_type = shape_def["shape"]
    points     = shape_def["points"]
    size       = shape_def["size"]

    if shape_type == "circle":
        center = points[0]
        radius_m = size
        return distance_m(center, pos) <= radius_m

    elif shape_type == "polygon":
        return point_in_polygon(pos, points)

    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")

def image_point_to_gps(pos, h, fov, heading, norm_x, norm_y, offset_u=0, offset_v=0):
    """
    Computes the ground GPS coordinate from a selected point in a downward-looking image,
    where the point is given as normalized coordinates (0 to 1, with 0 at left/top).

    Parameters:
      pos: dict with "lat", "lon", (and optionally "alt")
      h: altitude (in meters)
      fov: tuple (horizontal_fov, vertical_fov) in degrees (use your optimized FOV values)
      heading: heading in degrees (0° = north, increasing clockwise)
      norm_x, norm_y: normalized image coordinates (0 to 1, 0=left/top)
      offset_u: additional horizontal (u) offset in meters (from calibration)
      offset_v: additional vertical (v) offset in meters (from calibration)

    Returns:
      A GPS coordinate (as returned by vector_to_gps().json())
    """
    import math

    fov_h, fov_v = fov

    # Calculate ground half-extents based on effective (optimized) FOV.
    half_ground_width  = h * math.tan(math.radians(fov_h / 2))
    half_ground_height = h * math.tan(math.radians(fov_v / 2))
    
    # Compute offsets from the image center.
    # For normalized coordinates (0 to 1) the center is at 0.5.
    # Multiply by 2*half_extent to get the displacement in meters.
    # Then add the optimized offsets.
    u = (norm_x - 0.5) * 2 * half_ground_width + offset_u
    # Invert the y-axis because 0 is at the top.
    v = -(norm_y - 0.5) * 2 * half_ground_height + offset_v

    # Rotate the offset vector by the heading.
    heading_rad = math.radians(heading)
    east_offset  = u * math.cos(heading_rad) + v * math.sin(heading_rad)
    north_offset = -u * math.sin(heading_rad) + v * math.cos(heading_rad)
    
    # Compute the ground distance and azimuth.
    dist = math.hypot(east_offset, north_offset)
    az = (math.degrees(math.atan2(east_offset, north_offset)) + 360) % 360

    # Convert the computed vector into a GPS coordinate.
    current_position = GPSposition(pos["lat"], pos["lon"], 0)
    return vector_to_gps(current_position, dist, az)

def gps_to_image_point(cam_pos, gps, h, fov, heading, offset_u=0, offset_v=0):
    """
    Converts a GPS coordinate back into normalized image coordinates.
    
    This function is the inverse of image_point_to_gps(). Given a GPS coordinate (as computed
    by image_point_to_gps()) along with the camera parameters (position, altitude, FOV, heading,
    and calibration offsets), it computes the normalized (0 to 1) x,y coordinates corresponding 
    to that point in the image.
    
    Parameters:
      cam_pos: dict with "lat", "lon", (and optionally "alt") representing the camera's ground position
      gps: GPS coordinate (an object with attributes "lat" and "lon") to convert back into image space
      h: altitude in meters
      fov: tuple (horizontal_fov, vertical_fov) in degrees (use your optimized FOV values)
      heading: heading in degrees (0° = north, increasing clockwise)
      offset_u: additional horizontal (u) offset in meters (from calibration)
      offset_v: additional vertical (v) offset in meters (from calibration)
    
    Returns:
      A tuple (norm_x, norm_y) representing the normalized image coordinates (0 to 1, 0=left/top)
    """
    import math
    from geographiclib.geodesic import Geodesic

    fov_h, fov_v = fov

    # Compute the ground half-extents based on the FOV.
    half_ground_width  = h * math.tan(math.radians(fov_h / 2))
    half_ground_height = h * math.tan(math.radians(fov_v / 2))

    # Use geographiclib to compute the distance and bearing from the camera position to the GPS coordinate.
    geod = Geodesic.WGS84
    inv = geod.Inverse(cam_pos["lat"], cam_pos["lon"], gps.lat, gps.lon)
    dist = inv["s12"]
    az = inv["azi1"]
    az_rad = math.radians(az)
    
    # Compute the east and north offsets from the camera's ground position.
    east_offset  = dist * math.sin(az_rad)
    north_offset = dist * math.cos(az_rad)
    
    # Rotate the offsets back by the heading to obtain image frame offsets.
    heading_rad = math.radians(heading)
    u = east_offset * math.cos(heading_rad) - north_offset * math.sin(heading_rad)
    v = east_offset * math.sin(heading_rad) + north_offset * math.cos(heading_rad)

    # Remove calibration offsets.
    u_corr = u - offset_u
    v_corr = v - offset_v

    # Reverse the scaling from meters to normalized coordinates.
    norm_x = u_corr / (2 * half_ground_width) + 0.5
    norm_y = 0.5 - v_corr / (2 * half_ground_height)

    return norm_x, norm_y


def get_easting_letters(zone: int) -> str:
    """
    Return the valid easting letters for a given UTM zone in MGRS.
    The sequence cycles every 3 zones:
      - If zone % 3 == 1: use "ABCDEFGH"
      - If zone % 3 == 2: use "JKLMNPQR"
      - If zone % 3 == 0: use "STUVWXYZ"
    """
    mod = zone % 3
    if mod == 1:
        return "ABCDEFGH"
    elif mod == 2:
        return "JKLMNPQR"
    else:  # mod == 0
        return "STUVWXYZ"

# Northing letters are fixed for all zones.
NORTHING_LETTERS = "ABCDEFGHJKLMNPQRSTUV"  # 20 letters (I and O are omitted)

def parse_mgrs(mgrs_str: str, precision: int = None):
    """
    Parse an MGRS string into its components:
      - zone (int)
      - grid (2-letter string)
      - easting offset (int)
      - northing offset (int)
      - actual precision (number of digits per offset)

    The input MGRS string is assumed to be in the form:
         <zone><lat_band><grid><easting><northing>
    For example, "33RWH8359618530" means:
         Zone = 33, Band = R (ignored), Grid = WH, 
         Easting = 83596, Northing = 18530

    If precision is not provided or doesn’t match the numeric part,
    the numeric precision is auto-detected from the string.
    """
    mgrs_str = mgrs_str.strip().upper()
    # Extract the zone (one or two digits)
    i = 0
    while i < len(mgrs_str) and mgrs_str[i].isdigit():
        i += 1
    if i == 0:
        raise ValueError("Invalid MGRS string: no zone digits found")
    zone = int(mgrs_str[:i])
    
    # Next character: latitude band (ignored in the binary encoding)
    if i >= len(mgrs_str):
        raise ValueError("Invalid MGRS string: missing latitude band")
    band = mgrs_str[i]
    i += 1

    # Next two characters: the 100 km grid designator.
    if i + 1 >= len(mgrs_str):
        raise ValueError("Invalid MGRS string: missing 100km grid letters")
    grid = mgrs_str[i:i+2]
    i += 2

    # The remaining digits represent the easting and northing offsets.
    numeric_len = len(mgrs_str) - i
    if numeric_len % 2 != 0:
        raise ValueError("Numeric part length is not even.")
    detected_precision = numeric_len // 2
    if precision is None or precision != detected_precision:
        precision = detected_precision

    numeric = mgrs_str[i:]
    easting_str = numeric[:precision]
    northing_str = numeric[precision:]
    easting = int(easting_str)
    northing = int(northing_str)
    
    return zone, grid, easting, northing, precision

def encode_mgrs_binary(mgrs_str: str, precision: int = None) -> bytes:
    """
    Encode a full MGRS string into a compact binary format.
    
    Layout:
      • 1 byte for the UTM zone
      • 1 byte for the 100km grid designator (both letters encoded into one byte)
      • <bits_needed> bits for the easting offset
      • <bits_needed> bits for the northing offset
      
    bits_needed = ceil(log2(10^precision))
    
    The latitude band is not stored; a placeholder is used on decode.
    """
    zone, grid, easting, northing, actual_precision = parse_mgrs(mgrs_str, precision)
    
    # Determine grid letters based on zone.
    valid_easting_letters = get_easting_letters(zone)
    easting_letter = grid[0]
    northing_letter = grid[1]
    if easting_letter not in valid_easting_letters:
        raise ValueError(f"Invalid easting grid letter: {easting_letter} for zone {zone}, expected one of {valid_easting_letters}")
    try:
        easting_idx = valid_easting_letters.index(easting_letter)
    except ValueError:
        raise ValueError(f"Invalid easting grid letter: {easting_letter}")
    try:
        northing_idx = NORTHING_LETTERS.index(northing_letter)
    except ValueError:
        raise ValueError(f"Invalid northing grid letter: {northing_letter}")
    grid_index = easting_idx * len(NORTHING_LETTERS) + northing_idx

    max_value = 10 ** actual_precision
    bits_needed = math.ceil(math.log2(max_value))
    
    total_bits = 16 + 2 * bits_needed  # 8 bits for zone, 8 for grid_index
    total_bytes = (total_bits + 7) // 8
    
    combined = (zone & 0xFF) << (8 + 2 * bits_needed)
    combined |= (grid_index & 0xFF) << (2 * bits_needed)
    combined |= (easting & ((1 << bits_needed) - 1)) << bits_needed
    combined |= (northing & ((1 << bits_needed) - 1))
    
    return combined.to_bytes(total_bytes, byteorder='big')

def decode_mgrs_binary(data: bytes, precision: int) -> str:
    """
    Decode a full binary MGRS representation back into an MGRS string.
    
    Returned string is of the form:
         <zone><band_placeholder><grid><easting><northing>
    (The latitude band is not stored; "X" is used as a placeholder.)
    """
    bits_needed = math.ceil(math.log2(10 ** precision))
    total_bits = 16 + 2 * bits_needed
    total_bytes = (total_bits + 7) // 8
    if len(data) != total_bytes:
        raise ValueError("Invalid data length for the specified precision")
    
    combined = int.from_bytes(data, byteorder='big')
    
    northing_mask = (1 << bits_needed) - 1
    northing = combined & northing_mask
    combined //= (1 << bits_needed)
    easting = combined & northing_mask
    combined //= (1 << bits_needed)
    grid_index = combined & 0xFF
    combined //= (1 << 8)
    zone = combined & 0xFF
    
    valid_easting_letters = get_easting_letters(zone)
    num_northing_letters = len(NORTHING_LETTERS)
    easting_idx = grid_index // num_northing_letters
    northing_idx = grid_index % num_northing_letters
    try:
        easting_letter = valid_easting_letters[easting_idx]
    except IndexError:
        raise ValueError("Decoded easting letter index out of range")
    try:
        northing_letter = NORTHING_LETTERS[northing_idx]
    except IndexError:
        raise ValueError("Decoded northing letter index out of range")
    grid = easting_letter + northing_letter
    easting_str = str(easting).zfill(precision)
    northing_str = str(northing).zfill(precision)
    mgrs_str = f"{zone}X{grid}{easting_str}{northing_str}"
    return mgrs_str

def encode_relative_mgrs_binary(mgrs_str: str, precision: int = None, omit_zone: bool = False, omit_grid: bool = False) -> bytes:
    """
    Encode an MGRS string into a compact binary format while optionally omitting
    the GZD (zone) and/or the 100 km grid designator.

    Layout order:
      • [Zone] (8 bits)  -- if not omitted
      • [Grid index] (8 bits) -- if not omitted
      • [Easting offset] (<bits_needed> bits)
      • [Northing offset] (<bits_needed> bits)
    
    Parameters:
      mgrs_str: Full MGRS string.
      precision: Optionally force a specific precision; if None, auto-detected.
      omit_zone: If True, the 1-byte zone is omitted.
      omit_grid: If True, the 1-byte grid designator is omitted.
    
    Returns:
      A byte string representing the compact encoding.
    """
    zone, grid, easting, northing, actual_precision = parse_mgrs(mgrs_str, precision)
    
    valid_easting_letters = get_easting_letters(zone)
    easting_letter = grid[0]
    northing_letter = grid[1]
    if easting_letter not in valid_easting_letters:
        raise ValueError(f"Invalid easting grid letter: {easting_letter} for zone {zone}, expected one of {valid_easting_letters}")
    try:
        easting_idx = valid_easting_letters.index(easting_letter)
    except ValueError:
        raise ValueError(f"Invalid easting grid letter: {easting_letter}")
    try:
        northing_idx = NORTHING_LETTERS.index(northing_letter)
    except ValueError:
        raise ValueError(f"Invalid northing grid letter: {northing_letter}")
    grid_index = easting_idx * len(NORTHING_LETTERS) + northing_idx

    max_value = 10 ** actual_precision
    bits_needed = math.ceil(math.log2(max_value))
    
    total_bits = 2 * bits_needed
    if not omit_grid:
        total_bits += 8
    if not omit_zone:
        total_bits += 8
    total_bytes = (total_bits + 7) // 8

    combined = 0
    if not omit_zone:
        combined = (combined << 8) | (zone & 0xFF)
    if not omit_grid:
        combined = (combined << 8) | (grid_index & 0xFF)
    combined = (combined << bits_needed) | (easting & ((1 << bits_needed) - 1))
    combined = (combined << bits_needed) | (northing & ((1 << bits_needed) - 1))
    
    return combined.to_bytes(total_bytes, byteorder='big')

def decode_relative_mgrs_binary(data: bytes, precision: int, default_zone: int = None, default_grid: str = None,
                                omit_zone: bool = False, omit_grid: bool = False) -> str:
    """
    Decode the relative binary MGRS representation back into an MGRS string.
    
    If omit_zone is True, default_zone must be provided.
    If omit_grid is True, default_grid (a 2-letter string) must be provided.
    
    Returns an MGRS string of the form:
         <zone><band_placeholder><grid><easting><northing>
    (Using "X" as a placeholder for the latitude band.)
    """
    bits_needed = math.ceil(math.log2(10 ** precision))
    total_bits = 2 * bits_needed
    if not omit_grid:
        total_bits += 8
    if not omit_zone:
        total_bits += 8
    total_bytes = (total_bits + 7) // 8
    if len(data) != total_bytes:
        raise ValueError("Invalid data length for the specified precision and omitted fields")
    
    combined = int.from_bytes(data, byteorder='big')
    
    northing_mask = (1 << bits_needed) - 1
    northing = combined & northing_mask
    combined //= (1 << bits_needed)
    easting = combined & northing_mask
    combined //= (1 << bits_needed)
    
    if not omit_grid:
        grid_index = combined & 0xFF
        combined //= (1 << 8)
    else:
        grid_index = None
    if not omit_zone:
        zone = combined & 0xFF
    else:
        if default_zone is None:
            raise ValueError("Zone omitted but no default_zone provided")
        zone = default_zone

    if not omit_grid:
        valid_easting_letters = get_easting_letters(zone)
        num_northing_letters = len(NORTHING_LETTERS)
        easting_idx = grid_index // num_northing_letters
        northing_idx = grid_index % num_northing_letters
        try:
            easting_letter = valid_easting_letters[easting_idx]
        except IndexError:
            raise ValueError("Decoded easting letter index out of range")
        try:
            northing_letter = NORTHING_LETTERS[northing_idx]
        except IndexError:
            raise ValueError("Decoded northing letter index out of range")
        grid = easting_letter + northing_letter
    else:
        if default_grid is None or len(default_grid) != 2:
            raise ValueError("Grid omitted but no valid default_grid provided")
        grid = default_grid

    easting_str = str(easting).zfill(precision)
    northing_str = str(northing).zfill(precision)
    mgrs_str = f"{zone}X{grid}{easting_str}{northing_str}"
    return mgrs_str


# Usage Example
if __name__ == "__main__":
    a = GPSposition(lat=24.578524, lon=15.825613)

    # For comparison, show the lat/lon binary encoding.
    packed = struct.pack('<ii', int(a.lat * 1e7), int(a.lon * 1e7))
    print('Lat/Lon:', a)
    print('Lat/Lon binary (hex):', packed.hex(), "Length (bytes):", len(packed))
    print("-" * 40)

    mgrs_precision = 4
    full_mgrs = latlon_to_mgrs(a.lat, a.lon, precision=mgrs_precision)
    print("Full MGRS:", full_mgrs)
    # Encode full MGRS.
    binary_full = encode_mgrs_binary(full_mgrs)
    decoded_full = decode_mgrs_binary(binary_full, mgrs_precision)
    print("Full encoding -> binary (hex):", binary_full.hex(), "Length:", len(binary_full), "bytes")
    print("Decoded full MGRS:", decoded_full)
    print("-" * 40)
    
    # Now, encode relative MGRS omitting the GZD (zone) only.
    binary_relative_zone = encode_relative_mgrs_binary(full_mgrs, precision=mgrs_precision, omit_zone=True, omit_grid=False)
    # Extract default zone using parse_mgrs instead of split.
    zone_only, _, _, _, _ = parse_mgrs(full_mgrs)
    decoded_relative_zone = decode_relative_mgrs_binary(binary_relative_zone, mgrs_precision, default_zone=zone_only, omit_zone=True, omit_grid=False)
    print("Relative (omit zone) -> binary (hex):", binary_relative_zone.hex(), "Length:", len(binary_relative_zone), "bytes")
    print("Decoded relative (zone omitted):", decoded_relative_zone)
    print("-" * 40)
    
    # Finally, encode relative MGRS omitting both the GZD and the grid.
    binary_relative_both = encode_relative_mgrs_binary(full_mgrs, precision=mgrs_precision, omit_zone=True, omit_grid=True)
    # Extract default grid from the full MGRS using parse_mgrs.
    _, full_grid, _, _, _ = parse_mgrs(full_mgrs)
    decoded_relative_both = decode_relative_mgrs_binary(binary_relative_both, mgrs_precision, default_zone=zone_only, default_grid=full_grid, omit_zone=True, omit_grid=True)
    print("Relative (omit zone & grid) -> binary (hex):", binary_relative_both.hex(), "Length:", len(binary_relative_both), "bytes")
    print("Decoded relative (zone & grid omitted):", decoded_relative_both)
    print("-" * 40)