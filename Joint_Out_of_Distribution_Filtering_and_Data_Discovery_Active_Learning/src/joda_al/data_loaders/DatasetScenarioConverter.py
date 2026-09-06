import itertools
from typing import List, Tuple, Union

from torch.utils.data import Dataset


class DatasetScenarioConverter():

    @staticmethod
    def convert_task_to_pool_dataset(training_pool :Dataset, label_idx : List[int], unlabeled_idcs : List[int], validation_set:Dataset, test_set:Dataset) -> Tuple[Dataset, List[int], List[int], Dataset, Dataset]:
        """
                Converts the task dataset to a pool dataset.

                Args:
                    training_pool: The training pool dataset.
                    label_idx: List of labeled indices.
                    unlabeled_idcs: List of unlabeled indices.
                    validation_set: The validation dataset.
                    test_set: The test dataset.

                Returns:
                    Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
                """
        unlabeled_idx = list(itertools.chain(*unlabeled_idcs))
        if all(isinstance(idx, list) for idx in unlabeled_idx):
            print("list of lists detected")
            unlabeled_idx = list(itertools.chain(*unlabeled_idx))
        return training_pool, label_idx, unlabeled_idx, validation_set, test_set

    @staticmethod
    def convert_task_to_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set):
        """
        Converts the task dataset to a stream dataset.

        Args:
            training_pool: The training pool dataset.
            label_idx: List of labeled indices.
            unlabeled_idcs: List of unlabeled indices.
            validation_set: The validation dataset.
            test_set: The test dataset.

        Returns:
            Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
        """
        return training_pool, label_idx, unlabeled_idcs, validation_set, test_set

    def convert_task_to_flat_multi_stream_dataset(cls, training_pool: Dataset, label_idx: List[int], unlabeled_idcs: List[int], validation_set: Dataset, test_set: Dataset) -> Tuple[Dataset, List[int], List[int], Dataset, Dataset]:
        """
        Converts the task dataset to a flat multi-stream dataset.

        Args:
            training_pool (Dataset): The training pool dataset.
            label_idx (List[int]): List of labeled indices.
            unlabeled_idcs (List[int]): List of unlabeled indices.
            validation_set (Dataset): The validation dataset.
            test_set (Dataset): The test dataset.

        Returns:
            Tuple[Dataset, List[int], List[int], Dataset, Dataset]: Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
        """
        training_pool, label_idx, unlabeled_idcs, validation_set, test_set=cls.convert_task_to_multi_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)
        unlabeled_idx = list(itertools.chain(*unlabeled_idcs))
        return training_pool, label_idx, unlabeled_idx, validation_set, test_set

    @classmethod
    def convert_task_to_overlap_stream_dataset(cls, training_pool: Dataset, label_idx: List[int],
                                               unlabeled_idcs: List[int], validation_set: Dataset, test_set: Dataset,
                                               overlap: int = 50) -> Tuple[
        Dataset, List[int], List[int], Dataset, Dataset]:
        """
        Converts the task dataset to an overlap stream dataset.

        Args:
            training_pool (Dataset): The training pool dataset.
            label_idx (List[int]): List of labeled indices.
            unlabeled_idcs (List[int]): List of unlabeled indices.
            validation_set (Dataset): The validation dataset.
            test_set (Dataset): The test dataset.
            overlap (int, optional): The overlap percentage. Defaults to 50.

        Returns:
            Tuple[Dataset, List[int], List[int], Dataset, Dataset]: Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
        """
        training_pool, label_idx, unlabeled_idcs, validation_set, test_set= cls.convert_task_to_overlap_multi_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set,
                                                     overlap=overlap)
        unlabeled_idx = list(itertools.chain(*unlabeled_idcs))
        return training_pool, label_idx, unlabeled_idx, validation_set, test_set

    @staticmethod
    def convert_task_to_multi_stream_dataset(training_pool: Dataset, label_idx: List[int], unlabeled_idcs: Union[List[int],List[List[int]]],
                                             validation_set: Dataset, test_set: Dataset) -> Tuple[
        Dataset, List[int], List[List[int]], Dataset, Dataset]:
        """
        Converts the task dataset to a multi-stream dataset.

        Args:
            training_pool (Dataset): The training pool dataset.
            label_idx (List[int]): List of labeled indices.
            unlabeled_idcs (List[int]): List of unlabeled indices.
            validation_set (Dataset): The validation dataset.
            test_set (Dataset): The test dataset.

        Returns:
            Tuple[Dataset, List[int], List[int], Dataset, Dataset]: Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
        """
        if not isinstance(unlabeled_idcs[0][0], list):
            num_sub_streams = 10
            unlabeled_idcs_sub_stream = []
            for un_lab in unlabeled_idcs:
                sub_streams = []
                snippet_length = int(len(un_lab) / num_sub_streams)
                leftover = len(un_lab) % num_sub_streams
                for i in range(num_sub_streams):
                    sub_streams.append([un_lab[j] for j in range(i * snippet_length, (i + 1) * snippet_length)])
                if leftover > 0:
                    sub_streams[-1].extend(un_lab[-leftover:])
                unlabeled_idcs_sub_stream.append(sub_streams)
            unlabeled_idcs = unlabeled_idcs_sub_stream
        return training_pool, label_idx, unlabeled_idcs, validation_set, test_set

    @staticmethod
    def convert_task_to_multi_stream_dataset_alternating(training_pool: Dataset, label_idx: List[int],
                                                             unlabeled_idcs: Union[List[int],List[List[int]]], validation_set: Dataset,
                                                             test_set: Dataset) -> Tuple[Dataset, List[int], List[int], Dataset, Dataset]:
        """

            :param training_pool:
            :param label_idx:
            :param unlabeled_idcs:
            :param validation_set:
            :param test_set:
            :return:
            Converts the task dataset to a multi-stream dataset with alternating indices.

            Args:
                training_pool (Dataset): The training pool dataset.
                label_idx (List[int]): List of labeled indices.
                unlabeled_idcs (List[int]): List of unlabeled indices.
                validation_set (Dataset): The validation dataset.
                test_set (Dataset): The test dataset.

            Returns:
                Tuple[Dataset, List[int], List[int], Dataset, Dataset]: Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
        """
        if not isinstance(unlabeled_idcs[0][0], list):
            num_sub_streams = 10
            unlabeled_idcs_sub_stream = []
            for un_lab in unlabeled_idcs:
                sub_streams = []
                leftover = len(un_lab) % num_sub_streams
                for i in range(num_sub_streams):
                    sub_streams.append([un_lab[j] for j in range(i, len(un_lab)+i+1-num_sub_streams,num_sub_streams)])
                if leftover > 0:
                    sub_streams[-1].extend(un_lab[-leftover:])
                unlabeled_idcs_sub_stream.append(sub_streams)
            unlabeled_idcs = unlabeled_idcs_sub_stream
        return training_pool, label_idx, unlabeled_idcs, validation_set, test_set

    def convert_task_to_overlap_multi_stream_dataset(self, training_pool: Dataset, label_idx: List[int], unlabeled_idcs: List[List[int]], validation_set: Dataset, test_set: Dataset, overlap: int = 50) -> Tuple[Dataset, List[int], List[List[int]], Dataset, Dataset]:
        """
        Converts the task dataset to an overlap multi-stream dataset.

        Args:
            training_pool (Dataset): The training pool dataset.
            label_idx (List[int]): List of labeled indices.
            unlabeled_idcs (List[int]): List of unlabeled indices.
            validation_set (Dataset): The validation dataset.
            test_set (Dataset): The test dataset.
            overlap (int, optional): The overlap percentage. Defaults to 50.

        Returns:
            Tuple[Dataset, List[int], List[int], Dataset, Dataset]: Tuple containing the training pool, labeled indices, unlabeled indices, validation set, and test set.
        """
        training_pool, label_idx, unlabeled_idcs, validation_set, test_set = self.convert_task_to_multi_stream_dataset(training_pool, label_idx, unlabeled_idcs, validation_set, test_set)
        overlap_unlabeled_idc = []

        for un_labeled_streams in unlabeled_idcs:
            num_streams = len(un_labeled_streams)
            unlabeled_collection = []
            snippet_overlap = int(len(un_labeled_streams[0])*(overlap / 2 / 100)) # 25%
            first_snippet = un_labeled_streams[-1][-snippet_overlap:] + un_labeled_streams[0] + un_labeled_streams[1][:snippet_overlap]
            unlabeled_collection.append(first_snippet)
            for i in range(1, num_streams-1):
                snippet_overlap = int(len(un_labeled_streams[i]) * (overlap / 2 / 100))
                middle_snippet = un_labeled_streams[i-1][-snippet_overlap:] + un_labeled_streams[i] + un_labeled_streams[i+1][:snippet_overlap]
                unlabeled_collection.append(middle_snippet)
            snippet_overlap = int(len(un_labeled_streams[-1]) * (overlap / 2 / 100))
            last_snippet=un_labeled_streams[-2][-snippet_overlap:] + un_labeled_streams[-1] + un_labeled_streams[0][:snippet_overlap]
            unlabeled_collection.append(last_snippet)
            overlap_unlabeled_idc.append(unlabeled_collection)

        return training_pool, label_idx, overlap_unlabeled_idc, validation_set, test_set

    @staticmethod
    def convert_task_to_overlap_multi_stream_dataset_old(training_pool, label_idx, unlabeled_idcs, validation_set, test_set, overlap=50):
        if not isinstance(unlabeled_idcs[0][0], list):
            num_sub_streams = 10
            unlabeled_idcs_sub_stream = []
            for un_lab in unlabeled_idcs:
                sub_streams = []
                snippet_length = int(len(un_lab) / num_sub_streams)
                snippet_overlap = int((overlap/2/100)*snippet_length) #addtional 2 as symetric
                leftover = len(un_lab) % num_sub_streams
                for i in range(num_sub_streams):
                    sub_streams.append([un_lab[j] for j in range(i * snippet_length, (i + 1) * snippet_length)])
                if leftover > 0:
                    sub_streams[-1].extend(un_lab[-leftover:])
                inter_stream = [sub_streams[-1]] + sub_streams + [sub_streams[0]]
                overlap_sub_stream = [
                    inter_stream[i - 1][-snippet_overlap:] + inter_stream[i] + inter_stream[i + 1][:snippet_overlap] for i in
                    range(1, len(inter_stream) - 1)]
                unlabeled_idcs_sub_stream.append(overlap_sub_stream)
            unlabeled_idcs = unlabeled_idcs_sub_stream
        return training_pool, label_idx, unlabeled_idcs, validation_set, test_set
