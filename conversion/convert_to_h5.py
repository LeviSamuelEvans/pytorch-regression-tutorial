"""
============================
== ROOT to HDF5 Converter ==
============================

DESCRIPTION:
------------

    This python module provides functionality to convert datasets 
    from ROOT files into HDF5 files, with the specified structure
    described in the README.md file.

USAGE:
-----

    Please run python convert_to_h5.py -h to see the help message 
    and usage instructions.

"""

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import dataclasses
import os
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
from typing import Dict, List, Optional, Union, Any, Type
import argparse
import yaml
import sys
from colorama import Fore, Style
import glob
import time

# Rich stuff
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ------------------------------------------------------------
# RICH HELP FORMATTING
# ------------------------------------------------------------
class RichHelpFormatter(argparse.HelpFormatter):
    """A rich help formatter class that subclasses argparse.HelpFormatter."""

    def __init__(self, prog, indent_increment=2, max_help_position=24, width=None):
        super().__init__(prog, indent_increment, max_help_position, width)
        self.console = Console()

    def format_help(self):
        help_text = super().format_help()

        # format with rich
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))
        return ""


# ------------------------------------------------------------
# PROCESSORS
# ------------------------------------------------------------
# -> `ParticleProcessor`
# =======================
# The particle processor is used to process different particle
# objects that are present in an event, e.g. jets, electrons,
# muons, etc.

# -> `MetaDataProcessor`
# =======================
# The metadata processor is used to process metadata, e.g. event
# weights, pileup weights, etc.

# -> `GlobalPropertiesProcessor`
# =======================
# The global properties processor is used to process global
# properties, e.g. MET, HT, jet multiplicity, etc.

# -> `ClassificationProcessor`
# =======================
# The classification processor is used to process classification
# labels, e.g. event class, jet flavor, etc.

# -> `RegressionProcessor`
# =======================
# The regression processor is used to process regression
# targets, e.g. Higgs boson pT, etc.


# TODO
# - add processing for different HF processes
# - add methods to apply specified cuts to the data
# - parquet version
# - process files in batches and run in parallel
# ------------------------------------------------------------
@dataclasses.dataclass
class ParticleProcessor:
    """Base class for processing specific particle types.

    Note: one could substitute `particle` here with even
    a track, if one so wished. The class is designed to serve
    as a base class for objects with variable numbers of objects
    per event, hence the use of the MASK variable.

    Parameters:
    ----------
    name : str
        The name of the particle type.
    max_objects : int
        The maximum number of objects to process.
    variable_mapping : Dict[str, Union[str, List[str]]]
        The mapping of variables to the particle type.

    Methods:
    -------
    process_particles:
        Process particles of this type and store in the HDF5 file.

    populateMatrixWithEventData:
        Populate the matrix with event data.
    update_object_mask:
        Update the object mask.
    """

    name: str
    max_objects: int
    variable_mapping: Dict[str, Union[str, List[str]]]

    def process_particles(
        self, store: h5py.Group, data_dict: Dict[str, np.ndarray], num_events: int
    ) -> None:
        """Process particles of this type and store in the HDF5 file"""
        group = store.require_group(f"INPUTS/{self.name}")

        # Create mask for this particle type
        object_mask = np.zeros((num_events, self.max_objects), dtype=bool)

        for i in range(num_events):
            for variables in self.variable_mapping.items():

                # we need to first check if we are iterating over a list of variables
                # or just a single variable, this will alter the way we proceed
                if isinstance(variables[1], list):
                    for var in variables[1]:
                        # now, we check if the variable is in our data dictionary
                        if var in data_dict:
                            # if it is, we update the object mask
                            self.update_object_mask(data_dict, object_mask, i, var)
                        else:
                            # if it is not, we raise an error
                            raise ValueError(
                                f"Variable {var} not found in the data dictionary! Please check your variable mapping."
                            )

                # key-value pair is not linked to a list so we can proceed
                # without the instance check...
                else:
                    var_name = variables[1]
                    # same logic here as above, making sure we use the actual variable name
                    if var_name in data_dict:
                        self.update_object_mask(data_dict, object_mask, i, var_name)
                    else:
                        raise ValueError(
                            f"Variable {var_name} not found in the data dictionary! Please check your variable mapping."
                        )

        # create the updated object mask dataset in the HDF5 file
        group.create_dataset("MASK", data=object_mask, compression="gzip")

        # Create datasets for each variable
        for output_name, input_name in self.variable_mapping.items():

            # similar to the approach we took for the object mask, we need to check if
            # we are iterating over a list of variables or just a single variable
            if isinstance(input_name, list):
                for var in input_name:
                    # now, we check if the variable is in our data dictionary
                    if var in data_dict:

                        # create a matrix for this variable
                        matrix = np.zeros(
                            (num_events, self.max_objects), dtype=np.float32
                        )

                        # Now, we want to populate the matrix with our event data
                        # so we loop over the events
                        for i in range(num_events):
                            self.populateMatrixWithEventData(data_dict, i, var, matrix)

                        # now store the matrix in the HDF5 file
                        group.create_dataset(
                            output_name, data=matrix, compression="gzip"
                        )
                        # break here to avoid iterating over the same variable again
                        break
            # we have a single variable, so we can proceed in a more
            # conventional fashion here.
            else:
                if input_name in data_dict:

                    matrix = np.zeros((num_events, self.max_objects), dtype=np.float32)
                    for i in range(num_events):
                        self.populateMatrixWithEventData(
                            data_dict, i, input_name, matrix
                        )

                    # and now store again the matrix in the HDF5 file
                    group.create_dataset(output_name, data=matrix, compression="gzip")

    def populateMatrixWithEventData(
        self, data_dict: Dict[str, np.ndarray], i: int, var: str, matrix: np.ndarray
    ) -> None:
        """Populate the stored matrix with event data"""
        event_data = data_dict[var][i]
        n_objects = min(len(event_data), self.max_objects)
        if n_objects > 0:
            matrix[i, :n_objects] = event_data[:n_objects]

    def update_object_mask(
        self,
        data_dict: Dict[str, np.ndarray],
        object_mask: np.ndarray,
        i: int,
        var: str,
    ) -> None:
        """Update the object mask based on the event data"""
        event_data = data_dict[var][i]
        n_objects = min(len(event_data), self.max_objects)
        if n_objects > 0:
            object_mask[i, :n_objects] = True


# ------------------------------------------------------------
@dataclasses.dataclass
class MetaDataProcessor:
    """Base class for processing auxiliary data.

    Parameters:
    ----------
    variable_mapping : Dict[str, str]
        The mapping of variables to the metadata type.

    Methods:
    -------
    process_metadata:
        Process metadata and store in the HDF5 file.
    """

    variable_mapping: Dict[str, str]

    def process_metadata(
        self, store: h5py.Group, data_dict: Dict[str, np.ndarray], num_events: int
    ) -> None:
        """Process metadata and store in the HDF5 file"""
        group = store.require_group("INPUTS/METADATA")

        # create datsets for each metadata variable specified
        for output_name, input_name in self.variable_mapping.items():

            if input_name in data_dict:
                group.create_dataset(
                    output_name, data=data_dict[input_name], compression="gzip"
                )
            else:
                raise ValueError(
                    f"Variable {input_name} not found in the data dictionary!"
                )


# ------------------------------------------------------------
@dataclasses.dataclass
class GlobalPropertiesProcessor:
    """Class for processing global properties, like MET, HT, jet multiplicity, etc.

    Parameters:
    ----------
    variable_mapping : Dict[str, str]
        The mapping of variables to the global properties type.

    Methods:
    -------
    process_global_properties:
        Process global properties and store in the HDF5 file.

    """

    variable_mapping: Dict[str, str]

    def process_global_properties(
        self, store: h5py.Group, data_dict: Dict[str, np.ndarray], num_events: int
    ) -> None:
        """Process global properties and store in the HDF5 file"""
        group = store.require_group("INPUTS/GLOBAL_PROPERTIES")

        # create datasets for each global property
        for output_name, input_name in self.variable_mapping.items():
            if input_name in data_dict:
                # Global properties are event-level, so shape should be [num_events]
                group.create_dataset(
                    output_name, data=data_dict[input_name], compression="gzip"
                )
            else:
                raise ValueError(
                    f"Variable {input_name} not found in the data dictionary!"
                )


# ------------------------------------------------------------
@dataclasses.dataclass
class ClassificationProcessor:
    """Class for processing classification labels

    Parameters:
    ----------
    name : str
        The name of the classification type.
    class_map : Dict[str, int]
        The mapping of class names to integer values.
    class_variable : Optional[str]
        The optional variable to determine the class if filename-based classification doesn't work.

    Methods:
    -------
    process_classification:
        Process classification labels and store in the HDF5 file.

    Notes:
    -----
    This class is not neccesarily required, but is provided for convenience.
    The labels can easily be applied in ones favourite ML framework later on.

    """

    name: str
    class_map: Dict[str, int]
    # class_variable: Optional[str]

    def process_classification(
        self,
        store: h5py.Group,
        data_dict: Dict[str, np.ndarray],
        num_events: int,
        filename: Optional[str] = None,
    ) -> None:
        """Process classification labels and store in the HDF5 file"""
        group = store.require_group("TARGETS/CLASSIFICATION")

        # create classification label array
        labels = np.zeros(num_events, dtype=np.int32)

        # first, try determine class labels based on the filename
        if filename:
            class_id = None
            for class_name, id_value in self.class_map.items():
                if class_name in filename:
                    class_id = id_value
                    break

            if class_id is not None:
                # Apply the same class to all events
                labels.fill(class_id)
            else:
                print(f"Warning: No matching class found in filename {filename}")

        # Fall back to class_variable if filename-based classification doesn't work
        elif self.class_variable and self.class_variable in data_dict:
            for i in range(num_events):
                value = data_dict[self.class_variable][i]
                # Assign class label based on the value and class_map
                for class_name, class_id in self.class_map.items():
                    if class_name in str(value):
                        labels[i] = class_id
                        break
        else:
            print(
                f"Warning: Neither filename nor classification variable {self.class_variable} could be used for classification."
            )

        # Store the classification labels
        group.create_dataset(self.name, data=labels, compression="gzip")


# ------------------------------------------------------------
@dataclasses.dataclass
class RegressionProcessor:
    """Class for processing regression targets

    Parameters:
    ----------
    name : str
        The name of the regression type.
    variable_mapping : Dict[str, str]
        The mapping of variables to the regression type.

    Methods:
    -------
    process_regression:
        Process regression targets and store in the HDF5 file.
    """

    name: str
    variable_mapping: Dict[str, str]

    def process_regression(
        self, store: h5py.Group, data_dict: Dict[str, np.ndarray], num_events: int
    ) -> None:
        """Process regression targets and store in the HDF5 file"""
        group = store.require_group("TARGETS/REGRESSION")

        # Create datasets for each regression target
        for output_name, input_name in self.variable_mapping.items():
            if input_name in data_dict:
                # Regression targets are event-level, so shape should be [num_events]
                # for now at least anyway, could also add possibility of
                # object-level regression targets too?
                group.create_dataset(
                    output_name, data=data_dict[input_name], compression="gzip"
                )
            else:
                raise ValueError(
                    f"Variable {input_name} not found in the data dictionary!"
                )


# ------------------------------------------------------------
# CONVERTER
# ------------------------------------------------------------
# -> `Converter`
# The converter is the main class that orchestrates the
# conversion process. It is responsible for reading the root
# file(s), processing the data, and storing the processed data
# in a HDF5 file, with the specified structure.
# ------------------------------------------------------------
@dataclasses.dataclass
class Converter:
    """Main converter class that orchestrates the conversion process


    Parameters:
    ----------
    store_name : str
        The name of the store to write to.
    directory : str
        The directory of the root file to read from.
    variables : List[str]
        The variables to read from the root file.
    overwrite_enabled : bool
        Whether to overwrite the store if it already exists.
    store : Optional[h5py.File] = None
        The store to write to.
    particle_processors : Optional[List[ParticleProcessor]] = None
        The particle processors to use.
    global_properties_processor : Optional[GlobalPropertiesProcessor] = None
        The global properties processor to use.
    classification_processor : Optional[ClassificationProcessor] = None
        The classification processor to use.
    regression_processor : Optional[RegressionProcessor] = None
        The regression processor to use.
    metadata_processor : Optional[MetaDataProcessor] = None
        The metadata processor to use.

    Methods:
    -------
    __post_init__:
        Initialises the particle processors if they are not provided.
    __enter__:
        Enters the store for writing.
    __exit__:
        Exits the store for writing.
    getDataFrameFromRootfile:
        Gets the data from the root file and converts it to a pandas dataframe.
    check_var_exists:
        Checks if the variables exist in the root file.

    """

    store_name: str
    directory: str
    variables: List[str]
    overwrite_enabled: bool
    store: Optional[h5py.File] = None
    particle_processors: Optional[List[ParticleProcessor]] = None
    global_properties_processor: Optional[GlobalPropertiesProcessor] = None
    classification_processor: Optional[ClassificationProcessor] = None
    regression_processor: Optional[RegressionProcessor] = None
    metadata_processor: Optional[MetaDataProcessor] = None

    def __post_init__(self) -> None:
        """Initialise the particle processors if they are not provided"""
        if self.particle_processors is None:

            # Default processors with pre-made configurations
            # thanks to __post__init from dataclasses!
            self.particle_processors = [
                ParticleProcessor(
                    name="JETS",
                    max_objects=10,
                    variable_mapping={
                        "pt": "jet_pt",
                        "eta": "jet_eta",
                        "phi": "jet_phi",
                        "energy": "jet_e",
                        "btag": "jet_tagWeightBin_DL1r_Continuous",
                    },
                ),
                # configured for the single-lepton channel
                ParticleProcessor(
                    name="ELECTRONS",
                    max_objects=1,
                    variable_mapping={
                        "pt": "el_pt",
                        "eta": "el_eta",
                        "phi": "el_phi",
                        "energy": "el_e",
                        "charge": "el_charge",
                    },
                ),
                # configured for the single-lepton channel
                ParticleProcessor(
                    name="MUONS",
                    max_objects=1,
                    variable_mapping={
                        "pt": "mu_pt",
                        "eta": "mu_eta",
                        "phi": "mu_phi",
                        "energy": "mu_e",
                        "charge": "mu_charge",
                    },
                ),
            ]

            if self.global_properties_processor is None:
                self.global_properties_processor = GlobalPropertiesProcessor(
                    variable_mapping={
                        "met_met": "met_met",
                        "met_phi": "met_phi",
                        "nJets": "nJets",
                        "HT_all": "HT_all",
                    }
                )
            if self.classification_processor is None:
                self.classification_processor = ClassificationProcessor(
                    name="event_class",
                    class_map={
                        "ttH": 0,
                        "ttbb": 1,
                    },
                    # class_variable="process_id",
                )

            if self.regression_processor is None:
                self.regression_processor = RegressionProcessor(
                    name="higgs_pt",
                    variable_mapping={
                        "higgs_pt": "L2_Reco_higgs_pt",  # should be truth here but in te AFII samples we only have the reco so for testing purposes.
                    },
                )

            # can include any additional meadata here, useful for different event weights etc.
            if self.metadata_processor is None:
                self.metadata_processor = MetaDataProcessor(
                    variable_mapping={
                        "weight_mc": "weight_mc",
                        "weight_normalise": "weight_normalise",
                        "weight_jvt": "weight_jvt",
                        "weight_pileup": "weight_pileup",
                        "weight_leptonSF": "weight_leptonSF",
                    }
                )

    def __enter__(self) -> "Converter":
        """
        Enter the store for writing.
        """
        if self.overwrite_enabled and os.path.exists(self.store_name):
            os.remove(self.store_name)
            print(f"Overwriting enabled. Removed existing file: {self.store_name}")
        self.store = h5py.File(self.store_name, "w")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Exit the store for writing.
        """
        if self.store is not None:
            self.store.close()

    def get_root_filepaths(self):
        abs_directory = os.path.realpath(os.path.expanduser(self.directory))
        list_of_files = sorted(glob.glob(abs_directory + "/*.root"))
        return list_of_files

    def getDataFrameFromRootfile(
        self, tree_name: str, filepath: str, max_events: Optional[int]
    ) -> None:
        filename = os.path.basename(filepath)
        file_key = filename.replace(".", "_")

        # create file-specific groups in the store
        file_group = self.store.require_group(f"Files/{file_key}")

        if len(file_group.keys()) > 0:
            print(f"INFO: File {filename} already processed, skipping")
            return

        print(f"INFO: Opening ROOT file {filepath}")
        # print(
        #     f"INFO: Trying to extract the following variables:\n"
        #     + "\n".join(self.variables)
        # )
        tree = uproot.open(filepath)["nominal_Loose"]
        all_branches = tree.keys()

        # Check if the variables are in the tree
        self.check_var_exists(all_branches)

        # create arrays from the ROOT file
        data_dict = {}
        for var in self.variables:
            array = tree[var].array(library="np")
            if max_events is not None:
                array = array[:max_events]
            data_dict[var] = array

        # the number of events is based on the first variable in the variables list
        # this is because all variables should have the same number of events
        num_events = (
            max_events if max_events is not None else len(data_dict[self.variables[0]])
        )

        # Process particles
        for processor in self.particle_processors or []:
            processor.process_particles(file_group, data_dict, num_events)

        # Process global properties
        if self.global_properties_processor:
            self.global_properties_processor.process_global_properties(
                file_group, data_dict, num_events
            )

        # Process classification labels
        if self.classification_processor:
            self.classification_processor.process_classification(
                file_group, data_dict, num_events, filename=filename
            )

        # Process regression targets
        if self.regression_processor:
            self.regression_processor.process_regression(
                file_group, data_dict, num_events
            )

        # Process metadata
        if self.metadata_processor:
            self.metadata_processor.process_metadata(file_group, data_dict, num_events)

    def process_file(self, filepath, tree_name="nominal_Loose", max_events=None):
        """Process a single ROOT file"""
        self.getDataFrameFromRootfile(
            tree_name=tree_name, filepath=filepath, max_events=max_events
        )

    def process_all_files(self, tree_name="nominal_Loose", max_events=None):
        """
        Process all ROOT files in the specified directory with a progress bar.

        Args:
            tree_name (str): Name of the tree to process
            max_events (int, optional): Maximum number of events to process per file
        """
        filepaths = self.get_root_filepaths()
        for filepath in tqdm(filepaths, desc="Processing files", unit="files"):
            self.process_file(filepath, tree_name=tree_name, max_events=max_events)

    def check_var_exists(self, all_branches: List[str]) -> None:
        """Check if all requested variables exist in the ROOT file"""
        for var in self.variables:
            if var not in all_branches:
                raise ValueError(f"Variable {var} not found in the ROOT file")


if __name__ == "__main__":

    # ==============================================================================
    # COMMAND LINE ARGUMENTS
    def handle_command_line_args():
        """Parse and handle command-line arguments."""

        parser = argparse.ArgumentParser(
            description="A simple utility to convert ROOT files to HDF5 format.",
            formatter_class=RichHelpFormatter,
        )
        parser.add_argument(
            "-d",
            "--directory",
            help="Directory containing the ROOT files.",
            required=True,
        )
        parser.add_argument(
            "-s",
            "--store-name",
            help="Path for the output HDF5 store (default: store.h5).",
            default="store.h5",
        )
        parser.add_argument(
            "-v",
            "--variables",
            help="YAML file containing the list of variables to read from ROOT files \n"
            "Note: the variables must be in the format of 'feature: variable_name' \n"
            "where variable_name is the name of the variable in the ROOT file.",
            required=True,
        )
        parser.add_argument(
            "-O",
            "--overwrite",
            help="Overwrite existing store if it exists.",
            action="store_true",
            default=True,
        )
        parser.add_argument(
            "-n",
            "--num-events",
            type=int,
            help="Maximum number of events to process per file (default: process all events).",
            default=None,
        )
        parser.add_argument(
            "--max-jets",
            type=int,
            default=10,
            help="Maximum number of jets to process per event (default: 10).",
        )

        # TODO: implement this functionality to be able to order jets by some other variable
        # e.g. b-tagging score, pT, etc. If we are only, say, selecting 10 jets, we must
        # make sure we keep the jets with the highest b-tagging score, i.e. the jets that
        # are most interesting.
        parser.add_argument(
            "--order-jets",
            dest="order_jets",
            action="store_true",
            help="Order jets in each event by b-tagging score and pT (default: False).\n"
            "This is useful when using --max-jets option to avoid losing important jets.",
        )

        parser.set_defaults(order_jets=False)

        args = parser.parse_args()

        # Load variables from the YAML file
        with open(args.variables, "r") as file:
            yaml_content = yaml.safe_load(file)
            if "features" in yaml_content:
                args.variables = yaml_content["features"]
                print("Variables to be used from the YAML file:", args.variables)
            else:
                print("The YAML file does not contain a 'features' key.")
                sys.exit(1)

        if args.overwrite:
            while True:
                confirmation = input("--overwrite specified. Proceed? (y/n): ").lower()
                if confirmation in ["y", "yes"]:
                    break
                elif confirmation in ["n", "no"]:
                    sys.exit(1)
                else:
                    print("Invalid input. Answer 'yes' or 'no'.")

        return args

    args = handle_command_line_args()

    print(Fore.CYAN + "Starting the conversion process..." + Style.RESET_ALL)
    start_time = time.time()

    with Converter(
        store_name=args.store_name,
        directory=args.directory,
        overwrite_enabled=args.overwrite,
        variables=args.variables,
    ) as converter:

        print(Fore.GREEN + "Processing files..." + Style.RESET_ALL)
        converter.process_all_files(
            tree_name="nominal_Loose", max_events=args.num_events
        )
        print("INFO: Finished processing the ROOT file")

    print(Fore.YELLOW + "Conversion completed successfully!" + Style.RESET_ALL)
    print(Fore.GREEN + f"Total time taken: {time.time() - start_time:.2f} seconds" + Style.RESET_ALL)
    print(Fore.GREEN + "HDF5 file saved to: " + args.store_name + Style.RESET_ALL)
