import os
import numpy as np
import pandas as pd
from queue import PriorityQueue


class websiteOutputLoader:
    """This class load the data output we obtained by the website https://ped.fz-juelich.de/database/doku.php mentioned in the paper Tordeux 2019.
    It contains a class producing an output file with prefix "process_"
    """

    def __init__(self, dataset_folder_path, processed_output_path=None):
        self.dataset_folder_path = dataset_folder_path
        self.processed_output_path = processed_output_path
        if self.processed_output_path is None:
            self.processed_output_path = dataset_folder_path

    def loadData(self, dataset_name, numOfNeighbours, frame_rate=16, need_processing=False, contain_sk=True,
                 return_sk=True):
        """Load the dataset from we obtained from the website . Preprocess the dataset if it need processing.

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

        #checking if dataset flag should be processed (or it has been already) and building it accordingly
        if need_processing:
            self.process_rowdata(dataset_name, numOfNeighbours)
            dataset_path = os.path.join(self.processed_output_path, "processed_" + dataset_name)
        else:
            dataset_path = os.path.join(self.dataset_folder_path, dataset_name)

        tmpRawdata = np.loadtxt(dataset_path, delimiter=" ", skiprows=1)
        #masking data file into numpy arrays
        mask = np.repeat(True, numOfCols)
        mask[0] = False
        mask[1] = False
        if not contain_sk:
            mask[3] = False
        print(tmpRawdata)
        websiteRawdata = tmpRawdata[:, mask]
        # different return types based on the flag for returning sk or not (we have tw different NN to train)
        if return_sk:
            return websiteRawdata, tmpRawdata[:, 3]
        else:
            return websiteRawdata

    def process_rowdata(self, dataset_name, numOfNeighbours, frame_rate=16, need_processing=False, contain_sk=True,
                        return_sk=True):
        """process the data into the desired format, calculates sk, relative coordinates and saves it into the process_file

                Args:
                    dataset_name (str): name of the dataset file in the dataset folder.
                    numOfNeighbours (int): number of nearest neighbours
                    frame_rate(int): number of frames per second in the input data, needed to calculate the speed
                    need_processing (bool, optional): whether the given dataset need processing. Defaults to True.
                    contain_sk (bool, optional): whether the returned dataset contains the column mean spacing distance sk. Defaults to True.
                    return_sk (bool, optional): whether return the column of mean spacing distance as a single vector.

                Returns:
                    numpy.ndarray: the loaded and processed dataset
                """
        #building required paths
        dataset_path = os.path.join(self.dataset_folder_path, dataset_name)
        output_path = os.path.join(self.processed_output_path, "processed_" + dataset_name)

        #reading specific input format into np
        df = pd.read_csv(dataset_path, sep=" ", skiprows=1, usecols=[0, 1, 2, 3], comment="#")
        df.columns = ["pid", "frame", "x", "y"]

        # add speed and sort by frame

        # df_speed = pd.DataFrame(columns=['Speed'])
        df_speed = [-1]
        pid_old, frame_old, x_old, y_old = df.iloc[0]

        # get first line of row , initialize

        for index, row in df.iloc[1:].iterrows():
            pid, frame, x, y = row["pid"], row["frame"], row['x'], row['y']
            speed = self.calculateSpeed(pid, frame, x, y, pid_old, frame_old, x_old, y_old, frame_rate)

            df_speed.append(speed)

            pid_old, frame_old, x_old, y_old = pid, frame, x, y

        df['speed'] = df_speed

        columns_titles = ["frame", "pid", "speed", "x", "y"]
        df = df.reindex(columns=columns_titles)
        df = df.sort_values(by=['frame'])

        output = open(output_path, "w")
        output.write("timeStep pedestrianId speed sk kNearestNeighbors" + "\n")

        # specific time frame
        for i, timeGroup in df.groupby("frame"):

            for k, ped in timeGroup.iterrows():
                kneighbors = PriorityQueue(numOfNeighbours)

                for l, neighbor in timeGroup.iterrows():

                    if neighbor["pid"] == ped["pid"]:
                        continue
                    else:
                        if kneighbors.qsize() < numOfNeighbours:
                            distance = -((((ped["x"] - neighbor["x"]) ** 2 + (
                                    ped["y"] - neighbor["y"]) ** 2)) ** 0.5) / 100
                            kneighbors.put(
                                [distance, ((neighbor["x"] - ped["x"]) / 100, (neighbor["y"] - ped["y"]) / 100)])
                        else:
                            [x, (y, z)] = kneighbors.get()
                            distance = -((((ped["x"] - neighbor["x"]) ** 2 + (
                                    ped["y"] - neighbor["y"]) ** 2)) ** 0.5) / 100
                            if x < distance:
                                kneighbors.put(
                                    [distance, ((neighbor["x"] - ped["x"]) / 100, (neighbor["y"] - ped["y"]) / 100)])
                            else:
                                kneighbors.put([x, (y, z)])

                output.write(self.produceOutputLine(ped, kneighbors, numOfNeighbours))
        output.close()

        return df

    def calculateSpeed(self, pid, frame, x, y, pid_old, frame_old, x_old, y_old, frame_rate):
        """calculates the PedestrianSpeed according to the last two timesteps and the according positions, if they belong to the same pedestrian other

                        Args:
                            pid (int): current PedestrianID
                            frame (int): current frame number
                            x (float): current x Position
                            y (float): current y Position
                            pid_old (int): last timeSteps PedestrianID
                            frame_old (int): last timeSteps frame number
                            x_old (float): last timeSteps x Position
                            y_old (float): last timeSteps y Position
                            frame_rate (int): last timeSteps frame_rate

                        Returns:
                            float: speed in m/s if same pedestrian, else -1
                        """
        # 25frames per second, 100cm in meter
        if pid == pid_old:
            return ((((x - x_old) ** 2 + (y - y_old)) ** 2) ** 0.5) * (frame_rate / 100) / (frame - frame_old)

        else:
            return -1

    def produceOutputLine(self, ped, kneighbors, numOfNeighbours):
        """process the data into the desired format, calculates sk, relative coordinates and saves it into the process_file

                        Args:
                            ped panda.pdframe{pid, frame,x,y}: dataframe containing the needed pedestrian information id, frame number, x-position, y-position
                            kneighbors: PriorityQueue with elements:[x, (y, z)], where x is the distance to the pedestrian,
                                        y and z are the relative coordinates to the pedestrian, (distance is stored as -distance so that the heap is a desired min-heap)
                            numOfNeighbours (int): number of k Neighbors we are checking

                        Returns:
                            str: an output line for the dataset, if it has required attribute speed and numberofneighbors
                        """

        stringy = ""
        sk = 0
        size = kneighbors.qsize()
        # print(size)
        # We only want full lines don't we?
        if size != numOfNeighbours: return ""
        if ped["speed"] == -1: return ""

        for i in range(size):
            [x, (y, z)] = kneighbors.get()
            stringy = " " + str(y) + " " + str(z) + stringy
            sk += -x
        # if(size==0):
        #    return ""
        sk /= size
        return str(ped["frame"]) + " " + str(ped["pid"]) + " " + str(ped["speed"]) + " " + str(sk) + stringy + "\n"

    def mergeDataset(self, dataset_name_list, merged_dataset_name, numOfNeighbours, frame_rate=16,
                     need_processing=False,
                     contain_sk=True, return_sk=False):
        """Given a list of dataset files with the same output format from vadere, merge then into a single dataset file.

        Args:
            dataset_name_list (list): list of names of the dataset files
            merged_dataset_name (str): the name of the merged dataset file
            numOfNeighbours (int): number of nearest neighbors in the dataset
            need_processing (bool, optional): whether the datasets need to be processed. Defaults to True.
            contain_sk (bool, optional): whether the returned datasets contain the mean space distance column. Defaults to True.
            return_sk (bool, optional): whether return the column of mean spacing distance as a single vector. Defaults to False.

        Returns:
            ndarray: the loaded merged dataset as ndarray
        """

        numOfCols = 4 + 2 * numOfNeighbours

        merged_dataset_path = os.path.join(self.processed_output_path, merged_dataset_name)

        with open(merged_dataset_path, "w") as output:
            for i, dataset_name in enumerate(dataset_name_list):

                if need_processing:
                    self.process_rowdata(dataset_name, numOfNeighbours)
                    dataset_name = "processed_" + dataset_name
                    dataset_path = os.path.join(self.dataset_folder_path, dataset_name)
                else:
                    dataset_path = os.path.join(self.dataset_folder_path, dataset_name)

                with open(dataset_path, "r") as input:
                    if i == 0:
                        output.writelines(input.readlines())
                    else:
                        input.readline()
                        output.writelines(input.readlines())

        if return_sk:
            vadereRawdata, sk = self.loadData(merged_dataset_name, numOfNeighbours, need_processing=False,
                                              contain_sk=contain_sk, return_sk=return_sk)
            return vadereRawdata, sk
        else:
            vadereRawdata = self.loadData(merged_dataset_name, numOfNeighbours, need_processing=False,
                                          contain_sk=contain_sk, return_sk=return_sk)
            return vadereRawdata
