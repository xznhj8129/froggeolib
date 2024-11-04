from geographiclib.geodesic import Geodesic
import mgrs
import math
import geojson
import json

class GPSposition():
    def __init__(self, lat:float, lon:float, alt):
        self.lat = lat
        self.lon = lon
        self.alt = float(alt)
    def __str__(self):
        s = "Lattitude: {:.8f} Longitude: {:.8f} Altitude: {:.3f}".format(self.lat, self.lon, self.alt)
        return s
    def latlon(self):
        return (self.lat, self.lon)
    def json(self):
        return json.dumps({
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt
        })

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

def latlon_to_mgrs(latlon):
    milobj = mgrs.MGRS()
    return milobj.toMGRS(latlon.lat,latlon.lon,0)

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