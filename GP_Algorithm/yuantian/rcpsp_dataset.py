"""
File: rcpsp_dataset.py
Description: MRCPSP dataset loader for RCPSP problems.
Author: Yuan Tian
Created: 2026-01-05
Based on: discrete-optimization project (MIT License) https://github.com/airbus/discrete-optimization

Copyright (c) 2026 Yuan Tian
This file is licensed under the MIT License.
See the LICENSE file in the project root for full license information.
"""
import os
from abc import abstractmethod

from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel

class RCPSPDatabase:
    """
    RCPSPDatabase provides methods to access and load RCPSP problem file paths from MMLIB datasets.
    It supports MMLIB50, MMLIB100, and MMLIB+ datasets
    """
    DATA_FOLDER = r"discrete_optimization_data/mm/"
    # PSPLIB lives directly under discrete_optimization_data/, NOT under mm/
    # (unlike MMLIB) -- PSPLIB_DIR previously pointed at a nonexistent path
    # and every get_psplib_*_files() call silently returned [].
    PSPLIB_DIR = r"discrete_optimization_data/PSPLIB/"
    MMLIB_DIR = DATA_FOLDER + r"/MMLIB/"
    MMLIB_50_DIR = MMLIB_DIR + r"/MMLIB50/"
    MMLIB_100_DIR = MMLIB_DIR + r"/MMLIB100/"
    MMLIB_PLUS_DIR = MMLIB_DIR + r"/MMLIB+/"

    @classmethod
    def get_instance_list_from_txt(cls, txt_path: str) -> list[str]:
        """Get the instance list by specifying a text file. 

        Args:
            txt_path (str): _description_

        Returns:
            list[str]: List of instance file paths.
        """
        data_home = ""
        if "DATA_HOME" in os.environ:
            data_home = os.getenv("DATA_HOME")
        with open(txt_path, mode="r") as f:
            return [os.path.join(data_home, line.strip()) for line in f.readlines()]

    @classmethod
    def get_instance(cls, group_name: str, instance_name: str) -> RCPSPModel:
        """
        Load the in
        Args:
            group_name: Dataset group name, can be "MMLIB50", "MMLIB100", "MMLIB+"
            instance_name: Instance file name, e.g., "J501_1.mm"

        Returns:
            The RCPSPModel instance.
        """
        dir: str = None
        if group_name.upper() == "MMLIB50":
            dir = cls.MMLIB_50_DIR
        elif group_name.upper() == "MMLIB100":
            dir = cls.MMLIB_100_DIR
        elif group_name.upper() == "MMLIB+":
            dir = cls.MMLIB_PLUS_DIR
        else:
            raise ValueError(f"group name: {group_name} is not mathced.")
        fp: str = dir + instance_name
        if os.path.exists(fp):
            return parse_file(fp)
        else:
            raise FileNotFoundError(
                f"Instance {instance_name} not found in {group_name}"
            )

    @classmethod
    def get_all_MMLIB_PLUS_files(cls)-> list[str]:
        """
        Get all MMLIB+ file path.
        Returns:
            List of file paths.
        """
        return [
            os.path.join(cls.MMLIB_PLUS_DIR, file)
            for file in os.listdir(cls.MMLIB_PLUS_DIR)
            if os.path.isfile(os.path.join(cls.MMLIB_PLUS_DIR, file))
        ]

    @classmethod
    def get_all_MMLIB_PLUS_50_files(cls):
        """
        Get all file paths of instances with 50 activities in MMLIB+.

        Returns:
            List of file paths.
        """
        first = 1
        last = 324
        return [
            os.path.join(cls.MMLIB_PLUS_DIR, f"Jall{class_id}_{case_id}.mm")
            for class_id in range(first, last + 1)
            for case_id in range(1, 6)
        ]

    @classmethod
    def get_all_MMLIB_PLUS_100_files(cls):
        """
        Get all file paths of instances with 100 activities in MMLIB+.
        Returns:
            List of file paths.
        """
        first = 325
        last = 648
        return [
            os.path.join(cls.MMLIB_PLUS_DIR, f"Jall{class_id}_{case_id}.mm")
            for class_id in range(first, last + 1)
            for case_id in range(1, 6)
        ]

    @classmethod
    def get_all_MMLIB_50_files(cls):
        """
        Get all MMLIB50 file paths.
        Returns:
            List of file paths.
        """
        return [
            os.path.join(cls.MMLIB_50_DIR, file)
            for file in os.listdir(cls.MMLIB_50_DIR)
            if os.path.isfile(os.path.join(cls.MMLIB_50_DIR, file))
        ]

    @classmethod
    def get_all_MMLIB_100_files(cls):
        """
        Get all MMLIB100 file paths.
        Returns:
            List of file paths.
        """
        return [
            os.path.join(cls.MMLIB_100_DIR, file)
            for file in os.listdir(cls.MMLIB_100_DIR)
            if os.path.isfile(os.path.join(cls.MMLIB_100_DIR, file))
        ]

    @classmethod
    def get_some_MMLIB_50_each_class_files(
        cls, start: int = 1, end: int = 5, step: int = 1
    ) -> list[str]:
        """
        Get a list of file paths in MMLIB50 from each class within a specified range.
        A class contains 5 cases, indexed from 1 to 5.
        Class name is like J50{class_id}_{case_id}.mm, where class_id is from 1 to 108. case id is from 1 to 5.
        Args:
            start: The starting index of the case (inclusive).
            end: The ending index of the case (exclusive).
            step: The step size between cases.

        Returns:
            List of file paths.

        """
        if not all([1 <= start <= 5, 1 <= end <= 6, start <= end]):
            raise ValueError(
                "Start and end should be greater than 0 and less than 5, start should be less than end."
            )
        prefix: str = "J50"
        ext: str = ".mm"
        first: int = 1
        last: int = 108
        return [
            os.path.join(
                cls.MMLIB_50_DIR,
                "".join([prefix, str(class_id), "_", str(case_id), ext]),
            )
            for class_id in range(first, last + 1)
            for case_id in range(start, end, step)
        ]

    @classmethod
    def get_some_MMLIB_100_each_class_files(
        cls, start: int, end: int, step: int = 1
    ) -> list[str]:
        """
        Get a list of file paths in MMLIB100 from each class within a specified range.
        A class contains 5 cases, indexed from 1 to 5.
        Class name is like J100{class_id}_{case_id}.mm, where class_id is from 1 to 108. case id is from 1 to 5.
        Args:
            start: The starting index of the case (inclusive).
            end: The ending index of the case (exclusive).
            step: The step size between cases.

        Returns:
            List of file paths.
        """
        if not all([1 <= start <= 5, 1 <= end <= 6, start <= end]):
            raise ValueError(
                "Start and end should be greater than 0 and less than 5, start should be less than end."
            )
        prefix: str = "J100"
        ext: str = ".mm"
        first: int = 1
        last: int = 108
        return [
            os.path.join(
                cls.MMLIB_100_DIR,
                "".join([prefix, str(class_id), "_", str(case_id), ext]),
            )
            for class_id in range(first, last + 1)
            for case_id in range(start, end, step)
        ]

    @classmethod
    def get_some_MMLIB_PLUS_each_class_files(
        cls, start: int, end: int, step: int = 1
    ) -> list[str]:
        """
        Get a list of file paths in MMLIB+ from each class within a specified range.
        A class contains 5 cases, indexed from 1 to 5.
        Class name is like Jall{class_id}_{case_id}.mm, where class_id is from 1 to 648. case id is from 1 to 5.
        Args:
            start: The starting index of the case (inclusive).
            end: The ending index of the case (exclusive).
            step: The step size between cases.

        Returns:
            A list of file paths.
        """
        if not all([1 <= start <= 5, 1 <= end <= 6, start <= end]):
            raise ValueError(
                "Start and end should be greater than 0 and less than 5, start should be less than end."
            )
        prefix: str = "Jall"
        ext: str = ".mm"
        first: int = 1
        last: int = 648
        return [
            os.path.join(
                cls.MMLIB_PLUS_DIR,
                "".join([prefix, str(class_id), "_", str(case_id), ext]),
            )
            for class_id in range(first, last + 1)
            for case_id in range(start, end, step)
        ]

    @classmethod
    def get_some_MMLIB_PLUS_50_each_class_files(
        cls, start: int, end: int, step: int = 1
    ) -> list[str]:
        """
        Get a list of file paths in MMLIB+ with 50 activities from each class within a specified range.
        A class contains 5 cases, indexed from 1 to 5.
        Class name is like Jall{class_id}_{case_id}.mm, where class_id is from 1 to 324. case id is from 1 to 5.
        Args:
            start: The starting index of the case (inclusive).
            end: The ending index of the case (exclusive).
            step: The step size between cases.

        Returns:
            A list of file paths.
        """
        if not all([1 <= start <= 5, 1 <= end <= 6, start <= end]):
            raise ValueError(
                "Start and end should be greater than 0 and less than 5, start should be less than end."
            )
        prefix: str = "Jall"
        ext: str = ".mm"
        first: int = 1
        last: int = 324
        return [
            os.path.join(
                cls.MMLIB_PLUS_DIR,
                "".join([prefix, str(class_id), "_", str(case_id), ext]),
            )
            for class_id in range(first, last + 1)
            for case_id in range(start, end, step)
        ]

    @classmethod
    def get_some_MMLIB_PLUS_100_each_class_files(
        cls, start: int, end: int, step: int = 1
    ) -> list[str]:
        """
        Get a list of file paths in MMLIB+ with 100 activities from each class within a specified range.
        A class contains 5 cases, indexed from 1 to 5.
        Class name is like Jall{class_id}_{case_id}.mm, where class_id is from 325 to 648. case id is from 1 to 5.
        Args:
            start: The starting index of the case (inclusive).
            end: The ending index of the case (exclusive).
            step: The step size between cases.

        Returns:
            A list of file paths.
        """
        if not all([1 <= start <= 5, 1 <= end <= 6, start <= end]):
            raise ValueError(
                "Start and end should be greater than 0 and less than 5, start should be less than end."
            )
        prefix: str = "Jall"
        ext: str = ".mm"
        first: int = 325
        last: int = 648
        return [
            os.path.join(
                cls.MMLIB_PLUS_DIR,
                "".join([prefix, str(class_id), "_", str(case_id), ext]),
            )
            for class_id in range(first, last + 1)
            for case_id in range(start, end, step)
        ]

    # ── PSPLIB loaders ────────────────────────────────────────────────────
    # PSPLIB naming: {prefix}{class_id}_{instance_id}.mm
    # j20: MRCPSP 20-activity 3-mode, up to 64 classes × 10 instances = 554 files
    # j30: RCPSP  30-activity,         64 classes × 10 instances = 640 files
    # m5:  MRCPSP 5-mode,              64 classes × up to 10 instances = 558 files
    PSPLIB_J20_DIR = PSPLIB_DIR + "j20/"
    PSPLIB_J30_DIR = PSPLIB_DIR + "j30/"
    PSPLIB_M5_DIR  = PSPLIB_DIR + "m5/"

    @classmethod
    def get_psplib_j20_files(cls, start: int = 1, end: int = 5, step: int = 1) -> list[str]:
        """MRCPSP 20-activity 3-mode benchmark (j20). Up to 64 classes × 10 instances.

        Args:
            start: instance index within each class, inclusive (1–10).
            end:   instance index, exclusive.
        """
        files = []
        for c in range(1, 65):
            for i in range(start, end, step):
                fp = os.path.join(cls.PSPLIB_J20_DIR, f"j20{c}_{i}.mm")
                if os.path.exists(fp):
                    files.append(fp)
        return files

    @classmethod
    def get_psplib_j30_files(cls, start: int = 1, end: int = 5, step: int = 1) -> list[str]:
        """RCPSP 30-activity benchmark (j30). 64 classes, 10 instances each.

        Args:
            start: instance index within each class, inclusive (1–10).
            end:   instance index, exclusive.
        """
        return [
            os.path.join(cls.PSPLIB_J30_DIR, f"j30{c}_{i}.mm")
            for c in range(1, 65)
            for i in range(start, end, step)
        ]

    @classmethod
    def get_psplib_m5_files(cls, start: int = 1, end: int = 5, step: int = 1) -> list[str]:
        """MRCPSP 5-mode benchmark (m5). 64 classes, up to 10 instances each.

        Args:
            start: instance index within each class, inclusive (1–10).
            end:   instance index, exclusive.
        """
        files = []
        for c in range(1, 65):
            for i in range(start, end, step):
                fp = os.path.join(cls.PSPLIB_M5_DIR, f"m5{c}_{i}.mm")
                if os.path.exists(fp):
                    files.append(fp)
        return files

    @classmethod
    def get_some_MMLIB_each_class_instances(
        cls, group_name: str, start: int, end: int, step: int = 1
    ) -> list[RCPSPModel]:
        """
        Get some instances from MMLIB50, MMLIB100 or MMLIB+, from each class within a specified range.
        A class contains 5 cases, indexed from 1 to 5.
        Args:
            group_name: Dataset group name, can be "MMLIB50", "MMLIB100", "MMLIB+"
            start: The starting index of the case (inclusive).
            end: The ending index of the case (exclusive).
            step: The step size between cases.
        Returns:
            A list of RCPSPModel instances.
        """
        files: list[str] = []
        if group_name.upper() == "MMLIB50":
            files = cls.get_some_MMLIB_50_each_class_files(start, end, step)
        elif group_name.upper() == "MMLIB100":
            files = cls.get_some_MMLIB_100_each_class_files(start, end, step)
        elif group_name.upper() == "MMLIB+":
            files = cls.get_some_MMLIB_PLUS_each_class_files(start, end, step)
        else:
            raise ValueError(
                f"group_name can only be 'MMLIB50', 'MMLIB100' or 'MMLIB+' "
            )
        return [parse_file(f) for f in files]

    @classmethod
    def split_instances_validation_and_training(
        cls, filepath: str, training_files_path, validation_files_path: str
    ):
        """
        Split the instances listed in the given file into training and validation sets.
        For each problem class, one instance is randomly selected for validation, and the rest are used
        for training.
        Args:
            filepath: Path to the input file containing instance file paths.
            training_files_path: Path to the output file for training instances.
            validation_files_path: Path to the output file for validation instances.
        """
        def parse_info(instance_name: str) -> (int, int, int):
            _instance_name = str(instance_name)
            MMLIB_50_PREFIX = r"J50"
            MMLIB_100_PREFIX = r"J100"
            MMLIB_PLUS_PREFIX = r"Jall"
            if _instance_name.startswith(MMLIB_50_PREFIX):
                _cl = MMLIB_50_PREFIX
            elif _instance_name.startswith(MMLIB_100_PREFIX):
                _cl = MMLIB_100_PREFIX
            elif _instance_name.startswith(MMLIB_PLUS_PREFIX):
                _cl = MMLIB_PLUS_PREFIX
            _instance_name = _instance_name.removeprefix(_cl)
            parts = _instance_name.split("_")
            _pro = parts[0]
            _ins = parts[1]
            return _cl, _pro, _ins

        with open(filepath, mode="r") as input_file:
            lines = input_file.readlines()
        from pathlib import Path

        group: dict[int, list] = {}
        for line in lines:
            cl, pro, ins = parse_info(Path(line.strip()).stem)
            if pro not in group:
                group[pro] = []
            group[pro].append((line.strip()))

        # select one instance for validation
        validation_set = [sub.pop() for sub in group.values()]
        with open(validation_files_path, mode="w") as vp:
            for item in validation_set:
                vp.write(item + "\n")

        with open(training_files_path, mode="w") as tp:
            combined_cases = []
            for sub in group.values():
                combined_cases.extend(sub)
            for item in combined_cases:
                tp.write(item + "\n")


class DatasetProvider:
    def __init__(self, problem_set: list) -> None:
        self._source = problem_set

    @abstractmethod
    def next(self) -> list:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class StaticDatasetProvider(DatasetProvider):
    """Always return the same problem set no matter how many times `next()` calls.

    Args:
        DatasetProvider (_type_): _description_
    """

    def __init__(self, problem_set: list) -> None:
        super().__init__(problem_set)

    def next(self) -> list:
        return self._source


class EvenlyDividedDatasetProvider(DatasetProvider):
    """
    Split the problem_set into several groups, each group contains equal amount of problems.
    Each group are provided when `next()` calls.
    If divisibility is not possible, the final group will contain all remaining items(in strict mode).
    This provider is used for batch training. During the GP training, each group is used in one generation.
    """

    def __init__(
        self, problem_set: list[RCPSPModel], number_of_group: int, strict: bool = False
    ) -> None:
        """
        Args:
            problem_set (list[RCPSPModel]): All instances you want to use.

            number_of_group (int): number of groups you want to divide.

            strict (bool): `True` indicates that each group has strictly equal number of instances.
            `False` (default) indicates that the last group will contain all remaining instances.
        """
        super().__init__(problem_set)
        self._number_of_group: int = number_of_group
        step = len(problem_set) // self._number_of_group
        if not step:
            raise ValueError(
                f"Problem set is too small(size = {len(problem_set)}), even less than number of group{self._number_of_group})"
            )
        # precompute segments of problems
        self._segments = [
            self._source[g * step : (g + 1) * step]
            for g in range(self._number_of_group)
        ]
        # if not in strict mode, add the remaining items to the last group
        if not strict:
            self._segments[-1].extend(self._source[self._number_of_group * step :])
        self._cur_group_id: int = -1

    def next(self) -> list[RCPSPModel]:
        self._cur_group_id += 1
        if self._cur_group_id < self._number_of_group:
            return self._segments[self._cur_group_id]
        else:
            return self._segments[-1]

    def reset(self) -> None:
        self._cur_group_id = -1


class EmptyDataSetProvider(DatasetProvider):
    def __init__(self, problem_set: list = None) -> None:
        pass

    def next(self) -> list:
        return []


if __name__ == "__main__":
    # A simple test
    files = RCPSPDatabase.get_some_MMLIB_PLUS_50_each_class_files(1, 2)
    exists = [os.path.exists(f) for f in files]
    if all(exists):
        print("ok")



