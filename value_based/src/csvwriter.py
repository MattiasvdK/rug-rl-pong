import numpy as np
import os
from typing import List


class CSVWriter:
    """Writes results of the simulations to csv files.

    This class writes the results and hyperparameters to csv files.
    The hyperparameters for an agent are written to
    'data/<agent>_params.csv' while the results are written to
    'data/<agent>_results_<identifier>.csv'. Where <agent> is the name
    of the agent and <identifier> is the identifier of the agent.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        parameter_names: List[str] = None,
    ) -> None:
        """Initializes the CSVWriter object.

        Initializes the CSV writer and performs the needed overhead
        to see if the path is valid. If parameter names are passed
        it writes them to the parameter csv file.

        @param (str) data_dir: Path to the data directory.
        @param (str) model_name: Name of the model.
        @param (list[str]) parameter_names: List of parameter names.
        """

        assert os.path.exists(data_dir)
        self.data_path = data_dir
        self.model_name = model_name

        self.parameter_names = parameter_names
        self.param_path = os.path.join(self.data_path,
                                       f'{model_name}_params.csv')
        self.result_path = os.path.join(
            self.data_path,
            f'{model_name}_results'
            + ('.csv' if parameter_names is None else '')
        )

        if parameter_names is not None:
            with open(self.param_path, 'w') as file:
                file.write('identifier;')
                file.write(';'.join(parameter_names))
                file.write('\n')

    def write_results(
        self,
        result_array: List[np.ndarray],
        identifier: str = None,
    ) -> None:
        """Writes simulation results to csv.

        @param (np.ndarray) result_array: Array of simulation results.
        @param (str) identifier: Identifier of the used model.
        """
        # identifier is needed if there are parameters

        path = self.result_path if self.parameter_names is None \
            else (self.result_path + f'_{identifier}.csv')

        open_mode = 'a' if os.path.exists(path) else 'w'
        with open(path, open_mode) as file:
            for col in result_array:
                file.write(';'.join(list(map(str, col))))
                file.write('\n')

    def write_parameters(self, parameters: dict, identifier: str):
        assert self.parameter_names is not None

        to_write = [identifier]
        for parameter in self.parameter_names:
            to_write.append(str(parameters[parameter]))

        with open(self.param_path, 'a') as file:
            file.write(';'.join(to_write))
            file.write('\n')

