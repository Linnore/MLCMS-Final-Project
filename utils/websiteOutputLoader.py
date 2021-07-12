import os
import numpy as np
import pandas as pd
from queue import PriorityQueue


class websiteOutputLoader:
    """This class load the data output by "KNearestNeighborProcessor" from Vadere.
    """

    def __init__(self, dataset_folder_path, processed_output_path=None):
        self.dataset_folder_path = dataset_folder_path
        self.processed_output_path = processed_output_path
        if self.processed_output_path is None:
            self.processed_output_path = dataset_folder_path

    def processData(self, dataset_name):

        dataset_path = os.path.join(self.dataset_folder_path, dataset_name)
        output_path = os.path.join(self.processed_output_path, "processed_" + dataset_name)

        # dt = np.dtype[(('PedID', np.int), ('Frame', np.int), ('X', np.double), ('Y', np.double))]
        # dataset = np.loadtxt(dataset_path)
        df = pd.read_csv(dataset_path, sep=" ", header=None)
        df.columns = ["pid", "frame", "x", "y"]

        # add speed and sort by frame
        x_old, y_old = 0, 0
        pid_old, frame_old = 0, 0
        df_speed = pd.DataFrame(columns=['Speed'])
        for index, row in df.iterrows():
            pid, frame, x, y = row["pid"], row["frame"], row['x'], row['y']

            speed = self.calculateSpeed(pid, frame, x, y, pid_old, frame_old, x_old, y_old)

            df_speed = df_speed.append({'Speed': speed}, ignore_index=True)

            pid_old, frame_old, x_old, y_old = pid, frame, x, y

        df['speed'] = df_speed

        columns_titles = ["frame", "pid", "speed", "x", "y"]
        df = df.reindex(columns=columns_titles)
        df = df.sort_values(by=['frame'])

        output = open(output_path, "w")

        # specific time frame
        for i, timeGroup in df.groupby("frame"):
            #print(timeGroup)
            for k, ped in timeGroup.iterrows():
                kneighbors = PriorityQueue(10)

                for l, neighbor in timeGroup.iterrows():

                    if neighbor["pid"] == ped["pid"]:
                        continue
                    else:
                        if kneighbors.qsize() < kneighbors.maxsize:
                            distance = -((((ped["x"] - neighbor["x"]) ** 2 + (ped["y"] - neighbor["y"])**2)) ** 0.5)/100
                            kneighbors.put([distance, ((neighbor["x"] - ped["x"])/100, (neighbor["y"] - ped["y"])/100)])
                        else:
                            [x, (y, z)] = kneighbors.get()
                            distance = -((((ped["x"] - neighbor["x"]) ** 2 + (ped["y"] - neighbor["y"]) ** 2)) ** 0.5)/100
                            if x<distance:
                                kneighbors.put([distance, ((neighbor["x"] - ped["x"])/100, (neighbor["y"] - ped["y"])/100)])
                            else:
                                kneighbors.put([x, (y, z)])



                output.write(self.calculateSk(ped, kneighbors))
        output.close()

        return df

    def calculateSpeed(self, pid, frame, x, y, pid_old, frame_old, x_old, y_old):
        # 25frames per second, 100cm in meter
        if pid == pid_old:
            return ((((x - x_old) ** 2 + (y - y_old)) ** 2) ** 0.5) * (25 / 100) / (frame - frame_old)

        else:
            return -1

    def calculateSk(self, ped, kneighbors):
        stringy = ""
        sk = 0
        size = kneighbors.qsize()

        # We only want full lines don't we?
        if size != 10: return ""

        for i in range(size):
            [x, (y, z)] = kneighbors.get()
            stringy = str(y) + " " + str(z) + " "+stringy
            sk += -x
        #if(size==0):
        #    return ""
        sk /= size
        return str(ped["frame"]) + " " + str(ped["pid"]) + " " + str(ped["speed"]) + " " + str(sk) + " " + stringy + "\n"
