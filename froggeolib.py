from __future__ import annotations
# froggeolib
from geographiclib.geodesic import Geodesic
import mgrs
import math
import geojson
import json

# monkey patch since this is fucking retarded
_original_default = json.JSONEncoder().default

def _patched_default(self, obj):
    if hasattr(obj, 'json') and callable(obj.json):
        return obj.json()
    return _original_default(obj)

json.JSONEncoder.default = _patched_default

class PosObject():
    def __init__(self, lat:float, lon:float, alt):
        self.lat = lat
        self.lon = lon
        self.alt = float(alt)


class GPSposition():
    def __init__(self, lat:float, lon:float, alt:float, ce:float = 0.0, le:float = 0.0, json:dict={}, tup:tuple=()):
        if lat and lon:
            self.lat = lat
            self.lon = lon
            if alt:
                self.alt = float(alt)
            else:
                self.alt = 0.0
            self.ce = ce
            self.le = le
                
        elif json:
            self.lat = float(json["lat"])
            self.lon = float(json["lon"])
            self.alt = float(json["alt"])
            if "ce" in json:
                self.ce = ce
            if "le" in json:
                self.le = le
        elif tup:
            self.lat = float(tup[0])
            self.lon = float(tup[1])
            if len(tup)==3:
                self.alt = float(tup[3])
            if len(tup)==5:
                self.ce = float(tup[4])
                self.le = float(tup[5])
        else:
            self.lat = 0.0
            self.lon = 0.0
            self.alt = 0.0
            self.ce = 0.0
            self.le = 0.0

    def __str__(self):
        if self.ce>0:
            s = "Lattitude: {:.8f} Longitude: {:.8f} Altitude: {:.3f} CE: {:.1f} LE: {:.1f}".format(self.lat, self.lon, self.alt, self.ce, self.le)
        else:
            s = "Lattitude: {:.8f} Longitude: {:.8f} Altitude: {:.3f}".format(self.lat, self.lon, self.alt)
        return s

    def __json__(self):
        return self.json()
    def __dict__(self):
        return self.json()

    def latlon(self):
        return (self.lat, self.lon)

    def mgrs(self):
        milobj = mgrs.MGRS()
        return milobj.toMGRS(self.lat,self.lon)

    def json(self):
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "ce": self.ce,
            "le": self.le
        }

    #def __setstate__(self, state):
        # Called when the object is being deserialized
    #    self.__dict__.update(state)
    #    self.non_serializable_attribute = None 

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

def convert_geopaste(string):
    x = string.split(';')[0].split(':')[1].split(',')
    return GPSposition(float(x[0]),float(x[1]),float(0))

def latlon_to_mgrs(latlon,alt=0):
    milobj = mgrs.MGRS()
    return milobj.toMGRS(latlon.lat,latlon.lon, alt)

def mgrs_to_latlon(milgrid):
    milobj = mgrs.MGRS()
    return milobj.toLatLon(milgrid)

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
