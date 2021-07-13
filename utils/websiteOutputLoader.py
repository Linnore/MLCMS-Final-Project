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

    def loadData(self, dataset_name, numOfNeighbours, need_processing=True, contain_sk=True, return_sk=True):
        """Load the dataset from vadere. Preprocess the dataset if it need processing.

        Args:
            dataset_name (str): name of the dataset file in the dataset folder.
            numOfNeighbours (int): number of nearest neighbours
            need_processing (bool, optional): whether the given dataset need processing. Defaults to True.
            contain_sk (bool, optional): whether the returned dataset contains the column mean spacing distance sk. Defaults to True.
            return_sk (bool, optional): whether return the column of mean spacing distance as a single vector.

        Returns:
            numpy.ndarray: the loaded and processed dataset
        """
        numOfCols = 4 + 2 * numOfNeighbours

        if need_processing:
            self.process_rowdata(dataset_name,numOfNeighbours)
            dataset_path = os.path.join(self.processed_output_path, "processed_" + dataset_name)
        else:
            dataset_path = os.path.join(self.dataset_folder_path, dataset_name)

        tmpRawdata = np.loadtxt(dataset_path, delimiter=" ")
        mask = np.repeat(True, numOfCols)
        mask[0] = False
        mask[1] = False
        if not contain_sk:
            mask[3] = False


        websiteRawdata = tmpRawdata[:, mask]

        if return_sk:
            return websiteRawdata, tmpRawdata[:, 3]
        else:
            return websiteRawdata

    def process_rowdata(self,dataset_name,numOfNeighbours,need_processing=True,contain_sk=True,return_sk=True):
        dataset_path = os.path.join(self.dataset_folder_path, dataset_name)
        output_path = os.path.join(self.processed_output_path, "processed_" + dataset_name)

        # dt = np.dtype[(('PedID', np.int), ('Frame', np.int), ('X', np.double), ('Y', np.double))]
        # dataset = np.loadtxt(dataset_path)
        df = pd.read_csv(dataset_path, sep=" ", header=None)
        df.columns = ["pid", "frame", "x", "y"]

        # add speed and sort by frame


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
                kneighbors = PriorityQueue(numOfNeighbours)

                for l, neighbor in timeGroup.iterrows():

                    if neighbor["pid"] == ped["pid"]:
                        continue
                    else:
                        if kneighbors.qsize() < numOfNeighbours:
                            distance = -((((ped["x"] - neighbor["x"]) ** 2 + (ped["y"] - neighbor["y"])**2)) ** 0.5)/100
                            kneighbors.put([distance, ((neighbor["x"] - ped["x"])/100, (neighbor["y"] - ped["y"])/100)])
                        else:
                            [x, (y, z)] = kneighbors.get()
                            distance = -((((ped["x"] - neighbor["x"]) ** 2 + (ped["y"] - neighbor["y"]) ** 2)) ** 0.5)/100
                            if x<distance:
                                kneighbors.put([distance, ((neighbor["x"] - ped["x"])/100, (neighbor["y"] - ped["y"])/100)])
                            else:
                                kneighbors.put([x, (y, z)])



                output.write(self.calculateSk(ped, kneighbors,numOfNeighbours))
        output.close()

        return df

    def calculateSpeed(self, pid, frame, x, y, pid_old, frame_old, x_old, y_old):
        # 25frames per second, 100cm in meter
        if pid == pid_old:
            return ((((x - x_old) ** 2 + (y - y_old)) ** 2) ** 0.5) * (25 / 100) / (frame - frame_old)

        else:
            return -1

    def calculateSk(self, ped, kneighbors,numOfNeighbours):

        stringy = ""
        sk = 0
        size = kneighbors.qsize()
        #print(size)
        # We only want full lines don't we?
        if size != numOfNeighbours: return ""
        if ped["speed"] == -1: return ""

        for i in range(size):
            [x, (y, z)] = kneighbors.get()
            stringy = " "+str(y) + " " + str(z) + stringy
            sk += -x
        #if(size==0):
        #    return ""
        sk /= size
        return str(ped["frame"]) + " " + str(ped["pid"]) + " " + str(ped["speed"]) + " " + str(sk) + stringy + "\n"
