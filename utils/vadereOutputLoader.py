import os
import numpy as np


class vadereOutputLoader:
    """This class load the data output by "KNearestNeighborProcessor" from Vadere. 
    """

    def __init__(self, dataset_folder_path, processed_output_path=None):
        self.dataset_folder_path = dataset_folder_path
        self.processed_output_path = processed_output_path
        if self.processed_output_path is None:
            self.processed_output_path = dataset_folder_path

    def process_rowdata(self, dataset_name, numOfCols):
        """Given the data file directly generated by Vadere, this method will do the followings:
        Rows containing information of pedestrians outside the measurement area will be omitted.
        Rows containing information of pedestrians that have less than K neighbors will be omitted.

        The processed data would have the format:

            timeStep | pedID | pedVelocity | meanSpacingDistance (optional) | relative-coordinates-of-KNN (contains K*2 columns)

        The processed dataset will be output to the output folder with the name "processed_datasetname".

        Args:
            dataset_name (str): name of the dataset file in the dataset folder, e.g. "trial_data.txt"
            numOfCols (int): number of columns in the format mentioned above
        """
        dataset_path = os.path.join(self.dataset_folder_path, dataset_name)
        output_path = os.path.join(
            self.processed_output_path, "processed_"+dataset_name)

        rowID = -1
        with open(dataset_path, "r") as input, open(output_path, "w") as output:
            row = input.readline()
            output.write(row)
            while True:
                rowID += 1
                tmp_row = row.split()
                if len(tmp_row) == numOfCols:
                    v = float(tmp_row[2]) # velocity
                    if v>0:
                        output.write(row)
                row = input.readline()
                if not row:
                    break

    def loadData(self, dataset_name, numOfNeighbours, need_processing=True, contain_sk=True, return_sk=False):
        """Load the dataset from vadere. Preprocess the dataset if it need processing.

        Args:
            dataset_name (str): name of the dataset file in the dataset folder.
            numOfNeighbours (int): number of nearest neighbours
            need_processing (bool, optional): whether the given dataset need processing. Defaults to True.
            contain_sk (bool, optional): whether the returned dataset contains the column mean spacing distance sk. Defaults to True.
            return_sk (bool, optional): whether return the column of mean spacing distance as a single vector. Defaults to False.

        Returns:
            numpy.ndarray: the loaded and processed dataset
        """
        numOfCols = 4 + 2*numOfNeighbours

        if need_processing:
            self.process_rowdata(dataset_name, numOfCols)
            dataset_path = os.path.join(
                self.processed_output_path, "processed_"+dataset_name)
        else:
            dataset_path = os.path.join(self.dataset_folder_path, dataset_name)

        tmpRawdata = np.loadtxt(dataset_path, delimiter=" ", skiprows=1)

        mask = np.repeat(True, numOfCols)
        mask[0] = False
        mask[1] = False
        if not contain_sk:
            mask[3] = False
        
        vadereRawdata = tmpRawdata[:, mask]

        if return_sk:
            return vadereRawdata, tmpRawdata[:, 3]
        else:
            return vadereRawdata

    def mergeDataset(self, dataset_name_list, merged_dataset_name, numOfNeighbours, need_processing=True, contain_sk=True, return_sk=False):
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

        numOfCols = 4 + 2*numOfNeighbours
        
        merged_dataset_path = os.path.join(self.processed_output_path, merged_dataset_name)

        with open(merged_dataset_path, "w") as output:
            for i, dataset_name in enumerate(dataset_name_list):
                if need_processing:
                    self.process_rowdata(dataset_name, numOfCols)
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
            vadereRawdata, sk = self.loadData(merged_dataset_name, numOfNeighbours, need_processing=False, contain_sk=contain_sk, return_sk=return_sk)
            return vadereRawdata, sk
        else:
            vadereRawdata = self.loadData(merged_dataset_name, numOfNeighbours, need_processing=False, contain_sk=contain_sk, return_sk=return_sk)
            return vadereRawdata