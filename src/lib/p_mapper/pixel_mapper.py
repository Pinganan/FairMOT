import numpy as np
import cv2


class PixelMapper(object):
    """
    Create an object for converting pixels to geographic coordinates,
    using four points with known locations which form a quadrilteral in both planes
    Parameters
    ----------
    pixel_array : (69,2) shape numpy array
        The (x,y) pixel coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    lonlat_array : (69,2) shape numpy array
        The (lon, lat) coordinates corresponding to the top left, top right, bottom right, bottom left
        pixels of the known region
    """
    
    def __init__(self, port):
        assert port == 221 or port ==225, "Need port is 221 or 225"
        quad_coords_221 = {
            "lonlat": np.array([
                [1487,592],
                [1436,594],
                [1385,594],
                [1335,592],
                [1332,645],
                [1332,698],
                [1281,700],
                [1230,700],
                [1179,698],
                [1126,698],
                [1126,752],
                [1075,752],
                [1021,752],
                [970,752],
                [922,752],
                [868,750],
                [817,750],
                [765,747],
                [714,749],
                [714,804],
                [664,804],
                [611,804],
                [611,855],
                [557,853],
                [509,857],
                [505,803],
                [507,749],
                [455,749],
                [404,750],
                [353,749],
                [352,696],
                [352,647],
                [301,647],
                [300,592],
                [301,539],
                [300,489],
                [353,491],
                [402,491],
                [455,489],
                [509,491],
                [559,492],
                [662,489],
                [662,435],
                [714,437],
                [765,435],
                [815,437],
                [815,384],
                [815,330],
                [868,332],
                [922,334],
                [920,280],
                [970,278],
                [1023,278],
                [1073,278],
                [1127,278],
                [1178,278],
                [1179,226],
                [1179,176],
                [1231,175],
                [1281,174],
                [1331,175],
                [1331,227],
                [1382,226],
                [1437,227],
                [1434,281],
                [1485,280],
                [1488,332],
                [1485,385],
                [1487,539]      
            ]),
            "pixel": np.array([
                [624,625],
                [728,595],
                [828,566],
                [919,541],
                [856,506],
                [798,472],
                [879,450],
                [956,434],
                [1025,418],
                [1091,404],
                [1035,376],
                [1098,367],
                [1154,354],
                [1205,344],
                [1259,334],
                [1303,325],
                [1342,316],
                [1385,308],
                [1423,301],
                [1372,285],
                [1407,280],
                [1441,273],
                [1400,259],
                [1429,256],
                [1457,248],
                [1504,262],
                [1550,276],
                [1577,271],
                [1599,268],
                [1621,263],
                [1662,276],
                [1710,290],
                [1729,285],
                [1774,300],
                [1815,315],
                [1859,332],
                [1843,339],
                [1827,346],
                [1812,349],
                [1794,357],
                [1771,366],
                [1724,384],
                [1783,403],
                [1759,413],
                [1734,425],
                [1702,439],
                [1765,462],
                [1837,491],
                [1808,506],
                [1779,523],
                [1854,557],
                [1824,581],
                [1790,605],
                [1759,631],
                [1712,666],
                [1666,701],
                [1764,749],
                [1863,794],
                [1819,848],
                [1771,906],
                [1709,975],
                [1585,907],
                [1504,979],
                [1402,1052],
                [1274,964],
                [1147,1037],
                [1027,951],
                [922,870],
                [681,674]
            ])
        }
        quad_coords_225 = {
            "lonlat": np.array([
                [361, 459], # Upper left
                [1658, 432], # Upper right
                [1653, 1031], # Lower right
                [253, 1029], # Lower left
                [221, 620]
            ]),
            "pixel": np.array([
                [1886, 358], # Upper left
                [174, 1080], # Upper right
                [10, 449], # Lower right
                [1509, 204], # Lower left
                [1555, 727]
            ])
        }
        quad_coords = {221: quad_coords_221, 225: quad_coords_225}
        pixel_array = quad_coords[port]['pixel']
        lonlat_array = quad_coords[port]['lonlat']
        # print("\n")
        # print("getPerspectiveTransform: \n",cv2.getPerspectiveTransform(np.float32(pixel_array),np.float32(lonlat_array)))
        # print("findHomography: ",cv2.findHomography(np.float32(pixel_array),np.float32(lonlat_array)))
        self.M, mask1 = cv2.findHomography(np.float32(pixel_array),np.float32(lonlat_array))
        # print("\n")
        # print(self.M)
        # print("\n")
        self.invM, mask2 = cv2.findHomography(np.float32(lonlat_array),np.float32(pixel_array))
        
    def pixel_to_lonlat(self, pixel):
        """
        Convert a set of pixel coordinates to lon-lat coordinates
        Parameters
        ----------
        pixel : (N,2) numpy array or (x,y) tuple
            The (x,y) pixel coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (lon, lat) coordinates
        """
        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1,2)
        assert pixel.shape[1]==2, "Need (N,2) input array" 
        #print("or_pixel: \n",pixel)
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0],1))], axis=1)
        #print("pixel: \n",pixel)
        lonlat = np.dot(self.M,pixel.T)
        #print("lonlat: \n",lonlat)
        #print("new_lonlat: \n",lonlat[:2,:])
        #print("new_lonlat: \n",lonlat[2,:])
        #print("new_lonlat: \n",lonlat[:2,:]/lonlat[2,:])
        #print("new_lonlat: \n",(lonlat[:2,:]/lonlat[2,:]).T)
        return (lonlat[:2,:]/lonlat[2,:]).T
    
    def lonlat_to_pixel(self, lonlat):
        """
        Convert a set of lon-lat coordinates to pixel coordinates
        Parameters
        ----------
        lonlat : (N,2) numpy array or (x,y) tuple
            The (lon,lat) coordinates to be converted
        Returns
        -------
        (N,2) numpy array
            The corresponding (x, y) pixel coordinates
        """
        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1,2)
        assert lonlat.shape[1]==2, "Need (N,2) input array" 
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0],1))], axis=1)
        pixel = np.dot(self.invM,lonlat.T)
        
        return (pixel[:2,:]/pixel[2,:]).T
