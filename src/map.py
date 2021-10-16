
import numpy as np

from lib.tracker import matching

class MapTable(object):

    def __init__(self):
        self.id_counter = 0

        self.match = []
        self.match_a = []
        self.match_b = []

        self.single_a = {}
        self.single_b = {}

    def search_match(self, camera_result):
        camera1, camera2 = camera_result[0], camera_result[1]
        
        c1_id = [ t.track_id for t in camera1]
        u1 = [i for i in range(len(camera1))]
        c2_id = [ t.track_id for t in camera2]
        u2 = [i for i in range(len(camera2))]
        
        two_matching, two_id = [], []
        temp_c1, id_c1 = [], []
        temp_c2, id_c2 = [], []
        for i in range(len(self.match)):
            if self.match_a[i] in c1_id and self.match_b[i] in c2_id:
                two_matching.append( (c1_id.index(self.match_a[i]), c2_id.index(self.match_b[i])) )
                two_id.append(self.match[i])
            elif self.match_a[i] in c1_id:
                temp_c1.append(c1_id.index(self.match_a[i]))
                id_c1.append(self.match[i])
            elif self.match_b[i] in c2_id:
                temp_c2.append(c2_id.index(self.match_b[i]))
                id_c2.append(self.match[i])

        for i, j in two_matching:
            u1.remove(i)
            u2.remove(j)
        for i in temp_c1:
            u1.remove(i)
        for i in temp_c2:
            u2.remove(i)

        return two_matching, two_id, temp_c1, id_c1, temp_c2, id_c2, u1, u2


    def search_single(self, camera_result):
        trackers1, trackers2 = camera_result[0], camera_result[1]
        matrix = []
        for t1 in trackers1:
            temp = []
            for t2 in trackers2:
                cost = matching.features_embedding(t1.features, t2.features)
                cost = mid_2d_value(cost)
                map_distance = matching.cal_distance([t1.mapx, t1.mapy], [t2.mapx, t2.mapy])
                # inaccuracy in two camera
                map_distance = matching.distance_standard([map_distance], standard=150)
                cost += map_distance[0]
                temp.append(cost)
            if temp:
                matrix.append(temp)
        if matrix:
            print("OVA map features embedding matrix")
            for i in matrix: print(i)
        else:
            matrix = np.zeros((len(trackers1), len(trackers2)), dtype=np.float)
        matches, u1, u2 = matching.linear_assignment(np.asarray(matrix), thresh=1)     # thresh for table

        two_matching, two_id = [], []
        temp_c1, id_c1 = [], []
        temp_c2, id_c2 = [], []
        for i, j in matches:
            x = trackers1[i]
            y = trackers2[j]
            if x.track_id in self.single_a and y.track_id in self.single_b:
                self.add_match(x.track_id, y.track_id, self.single_a[x.track_id])
                del self.single_a[x.track_id]
                del self.single_b[y.track_id]
            elif x.track_id in self.single_a:
                self.add_match(x.track_id, y.track_id, self.single_a[x.track_id])
                del self.single_a[x.track_id]
            elif y.track_id in self.single_b:
                self.add_match(x.track_id, y.track_id, self.single_b[y.track_id])
                del self.single_b[y.track_id]
            else:
                self.add_match(x.track_id, y.track_id, x.track_id)
            two_matching.append((i, j))
            two_id.append(self.match[-1])

        for i in u1:
            x = trackers1[i]
            if x.track_id not in self.single_a:
                self.single_a[x.track_id] = 10000 + x.track_id
                #self.single_a[x.track_id] = "c_" + str(x.track_id)
            id_c1.append(self.single_a[x.track_id])
            temp_c1.append(i)
        for i in u2:
            x = trackers2[i]
            if x.track_id not in self.single_b:
                self.single_b[x.track_id] = 20000 + x.track_id
                #self.single_b[x.track_id] = "c2_" + str(x.track_id)
            id_c2.append(self.single_b[x.track_id])
            temp_c2.append(i)

        return two_matching, two_id, temp_c1, id_c1, temp_c2, id_c2


    def add_match(self, a_id, b_id, nid):
        self.match.append(nid)
        self.match_a.append(a_id)
        self.match_b.append(b_id)
        self.id_counter += 1


def mid_2d_value(cost):
    nlist = flatten(cost)
    nlist.sort()
    return nlist[int(len(nlist)/2)]

def flatten(cost):
    new_list = []
    for i in cost:
        for j in i:
            new_list.append(j)
    return new_list

