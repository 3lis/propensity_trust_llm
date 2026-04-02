"""
#####################################################################################################################

    HURST project - 2024

    Configuration of parameters from file and command line

#####################################################################################################################
"""

import  os
import  torch
from    argparse        import ArgumentParser


class Config( object ):
    """
    Parameters accepted by the software:
    (many parameters can be ser in the configuration file as well as with command line flags)

    AUGMENTATION            [bool] print a description of all augmentation codes
    CONFIG                  [str] name of configuration file (without path nor extension) (DEFAULT=None)
    DEBUG                   [bool] debug mode: print prompts only, do not call OpenAI
    LIST                    [bool] list information about models, files, fine-tuning
    MAXTOKENS               [int] maximum number of tokens (DEFAULT=None)
    MODEL                   [int] index in the list of possible models (DEFAULT=0)
    NRETURNS                [int] number of return sequences (DEFAULT=None)
    RECOVER:                [bool] recover from a previous aborted execution
    TEST                    [bool] test a model with a simple prompt
    VERBOSE                 [int] write additional information, -v standard, -vv for debugging

    agents                  [list] with names of the agents
    augmentation            [int] level of augmentation of the language models used by the agents
    easiness                [float] global level of easiness of tasks
    extra_memory            [bool] use the extended short term memory for trust belief per tasks
    f_dialog                [str] filname of json file with dialogs
    f_agents                [str] filname of json file with all agents in the simulation
    init_dialog             [list] titles of initial dialogues
    model_id                [int] index in the list of possible models (overwritten by MODEL)
    n_returns               [int] number of return sequences (overwritten by NRETURNS)
    max_tokens              [int] maximum number of tokens (overwritten by MAXTOKENS)
    memory_dialogs          [list] two dialogues for memory, the first for boostrap
    memory_lag              [int] the maximum memory lag for remembery past agents' events
    outcome_fn              [str] the type of computation of the outcome of a task ("digital", "float", "disjoint" )
    randomness              [float] amount of randomness when computing task outcome
    random_task             [bool] randomize the selection of next task during runs
    scenario                [str] name of json file with the scenario (no extension)
    steps                   [int] simulation steps
    tasks                   [list] with names of the tasks
    temperature             [float] sampling temperature during completion (default=1.0)
    top_p                   [int] probability mass of tokens generated in completion (default=1)
    trust_mode              [str] how trust is managed ("txt", "num")
    """

    def load_from_line( self, line_kwargs ):
        """
        Load parameters from command line arguments

        params:
            line_kwargs:        [dict] parameteres read from arguments passed in command line
        """
        for key, value in line_kwargs.items():
            setattr( self, key, value )


    def load_from_file( self, file_kwargs ):
        """
        Load parameters from a python file.
        Check the correctness of parameteres, set defaults.

        params:
            file_kwargs:        [dict] parameteres coming from a python module (file)
        """
        for key, value in file_kwargs.items():
            setattr( self, key, value )

        if not hasattr( self, 'trust_mode' ):
            self.trust_mode         = "txt"
        if not hasattr( self, 'agents' ):
            self.agents             = None
        if not hasattr( self, 'tasks' ):
            self.tasks              = None
        if not hasattr( self, 'memory_lag' ):
            self.memory_lag         = 10
        if not hasattr( self, 'n_returns' ):
            self.n_returns          = 1
        if not hasattr( self, 'max_tokens' ):
            self.max_tokens         = 20
        if not hasattr( self, 'top_p' ):
            self.top_p              = 1
        if not hasattr( self, 'temperature' ):
            self.temperature        = 1.0
        if not hasattr( self, 'augmentation' ):
            self.augmentation       = 2
        if not hasattr( self, 'outcome_fn' ):
            self.outcome_fn         = "float"


    def __str__( self ):
        """
        Visualize the list of all parameters

        return:     [str]
        """
        s   = ''
        d   = self.__dict__

        for k in d:
            if isinstance( d[ k ], dict ):
                s   += f"{k}:\n"
                for j in d[ k ]:
                    s   += f"{'':5}{j:<30}{d[ k ][ j ]}\n"
            else:
                s   += f"{k:<35}{d[ k ]}\n"
        return s


def read_args():
    """
    Parse the command-line arguments defined by flags
    
    return:         [dict] key = name of parameter, value = value of parameter
    """
    parser      = ArgumentParser()

    parser.add_argument(
            '-a',
            '--augmentation',
            action          = 'store_true',
            dest            = 'AUGMENTATION',
            help            = "show information about augmentation code"
    )
    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            default         = None,
            help            = "Name of configuration file (without path nor extension)"
    )
    parser.add_argument(
            '-D',
            '--debug',
            action          = 'store_true',
            dest            = 'DEBUG',
            help            = "Debug mode: print prompts only, do not call OpenAI"
    )
    parser.add_argument(
            '-L',
            '--list',
            action          = 'store_true',
            dest            = 'LIST',
            help            = "List information about models, files, fine-tuning"
    )
    parser.add_argument(
            '-m',
            '--model',
            action          = 'store',
            dest            = 'MODEL',
            type            = int,
            default         = None,
            help            = "Index in the list of possible models (default=0) (-1 to print all)",
    )
    parser.add_argument(
            '-M',
            '--maxreturns',
            action          = 'store',
            dest            = 'MAXTOKENS',
            type            = int,
            default         = None,
            help            = "Maximum number of tokens (default=500)",
    )
    parser.add_argument(
            '-n',
            '--nreturns',
            action          = 'store',
            dest            = 'NRETURNS',
            type            = int,
            default         = None,
            help            = "Number of return sequences (default=1)",
    )
# recovery has not been maintained - it would be still useful, however it requires quite
# a lot of work due to the differrent entities to save for all the augmentations
#   parser.add_argument(
#           '-r',
#           '--recover',
#           action          = 'store_true',
#           dest            = 'RECOVER',
#           help            = "Recover from the previous failed execution"
#   )
    parser.add_argument(
            '-R',
            '--redirect',
            action          = 'store_true',
            dest            = 'REDIRECT',
            help            = "Redirect stderr and stdout to log files"
    )
    parser.add_argument(
            '-t',
            '--test',
            action          = 'store_true',
            dest            = 'TEST',
            help            = "test a model with a simple prompt"
    )
    parser.add_argument(
            '-v',
            '--verbose',
            action          = 'count',
            dest            = 'VERBOSE',
            default         = 0,
            help            = "Write additional information, -v standard, -vv more"
    )
    return vars( parser.parse_args() )


