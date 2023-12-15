import pandas as pd
import numpy as np

from utils.ftl_log import LOGGER


class FTLDataLoader:
    """
    input the path of data, get these variables:
    data_inst: pandas Dataframe of the data
    id_index_map: the mapping of data id to the index in data_inst or data_matrix
    data_matrix: the numpy matrix of data, column 'id' and 'y' (if it has) has been deleted
    labels: numpy array of the 'y' column (if exists)
    """

    def __init__(self, file_name):
        self.data_frame: pd.DataFrame = self.__load_data(file_name)
        self.id_index_map: dict = dict(
            zip(self.data_frame["id"], range(len(self.data_frame["id"])))
        )
        if "y" in self.data_frame.columns:
            self.labels: np.array = np.array(self.data_frame["y"])
        self.data_matrix = np.matrix(
            self.data_frame[self.data_frame.columns.difference(["id", "y"])]
        )
        # translate data_matrix to numpy.ndarray
        self.data_matrix = self.data_matrix.A

    def __load_data(self, file_name):
        """
        :param file_name: file path
        :return: a data frame in pandas for the data
        :action: doing label check and return data frame
        """
        try:
            data_frame: pd.DataFrame = pd.read_csv(file_name)
            assert "id" in data_frame.columns, "dataset must contains 'id' column"

            if "y" in data_frame.columns:
                self.__label_check(data_frame)

            return data_frame

        except FileNotFoundError as e:
            LOGGER.error(
                f"file not found, please check the path: '{file_name}', only supports .csv file now"
            )
            raise e

    def __label_check(self, data_frame):
        label_set = set(data_frame["y"])
        assert (
            len(label_set) == 2
        ), f"only supprots 2 classes of labels, but {label_set} is offered"
        if label_set != {-1, +1}:
            label_list = list(label_set)
            LOGGER.debug(
                f"mapping label value {label_list[0]} to -1, {label_list[1]} to +1"
            )
            value_map = {label_list[0]: -1, label_list[1]: 1}
            for i, _ in enumerate(data_frame["y"]):
                data_frame.loc[i, "y"] = value_map[data_frame.loc[i, "y"]]
