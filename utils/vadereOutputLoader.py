import os
import numpy as np


class vadereOutputLoader:
    def __init__(self, dataset_folder_path, processed_output_path=None):
        self.dataset_folder_path = dataset_folder_path
        self.processed_output_path = processed_output_path
        if self.processed_output_path is None:
            self.processed_output_path = dataset_folder_path
    
    def process_rowdata(self, dataset_name, numOfCols):
        dataset_path = os.path.join(self.dataset_folder_path, dataset_name)
        output_path = os.path.join(self.processed_output_path, "processed_"+dataset_name)

        rowID = -1
        with open(dataset_path, "r") as input, open(output_path, "w") as output:
            row = input.readline()
            output.write(row)
            while True:
                rowID += 1
                tmp_row = row.split()
                if len(tmp_row) == numOfCols:
                    output.write(row)
                row = input.readline()
                if not row:
                    break

    def loadData(self, dataset_name, numOfNeighbours, need_processing = True, contain_sk = True):
        numOfCols = 4 + 2*numOfNeighbours
        if not contain_sk:
            numOfCols -= 1

        if need_processing:
            self.process_rowdata(dataset_name, numOfCols)
            dataset_path = os.path.join(self.processed_output_path, "processed_"+dataset_name)
        else:
            dataset_path = os.path.join(self.dataset_folder_path, dataset_name)

        vadereRawdata = np.loadtxt(dataset_path, delimiter=" ", skiprows=1)
        

        return vadereRawdata