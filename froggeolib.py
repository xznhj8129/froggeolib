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
    def __init__(self, lat:float, lon:float, alt):
        self.lat = lat
        self.lon = lon
        self.alt = float(alt)

    def __str__(self):
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
            "alt": self.alt
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

def vector_to_gps_air(latlon, az, ang): #only valid if both points are at same altitude
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
