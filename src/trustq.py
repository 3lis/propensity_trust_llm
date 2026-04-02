"""
#####################################################################################################################

    trust propensity project - 2025

    test models with Frazier's trust propensity scale

#####################################################################################################################
"""

import  os
import  sys
import  json5
import  time
import  datetime
import  numpy       as np

import  load_cnfg
import  agent
import  simulation
import  lm
import  complete

from    models      import models, models_interface, models_short_name

DO_NOTHING              = False                 # for debugging
DEBUG                   = False                 # temporary specific debugging

now_time                = None
frmt_response           = "%y-%m-%d_%H-%M-%S"   # datetime format for response filenames
frmt_trainset           = "%y-%m-%d-%H"         # datetime format for training set filenames

cnfg                    = None                  # configuration object

dir_res             = '../res'
dir_data            = '../data'
dir_current         = None
data_fname          = "frazier"                                 # filename of the benchmark
f_dialog            = "dialogs"                                 # filename of the dialogs
title               = "prologue_propensity_boolean"             # default dialog to use
log_runs            = 'runs.log'
log_err             = "err.log"
log_msg             = "msg.log"

repetitions         = 10                                        # how many repetitions per item


def init_dirs():
    """ -------------------------------------------------------------------------------------------------------------
    Set paths to directories where to save the execution
    ------------------------------------------------------------------------------------------------------------- """
    global dir_current
    global log_runs, log_err, log_msg

    dir_current     = os.path.join( dir_res, now_time )
    while os.path.isdir( dir_current ):
        if cnfg.VERBOSE:
            print( f"Warning: {dir_current} already existing, creating with one more second" )
        sec         = int( dir_current[ -2 : ] )
        sec         += 1
        dir_current = f"{dir_current[ : -2 ]}{sec:02d}"

    os.makedirs( dir_current )

    log_runs        = os.path.join( dir_current, log_runs )
    log_err         = os.path.join( dir_current, log_err )
    log_msg         = os.path.join( dir_current, log_msg )


def init_cnfg():
    """
    Set global parameters from command line and python configuration file
    Execute this function before init_dirs() and before executions that involve simulation
    """
    global cnfg, now_time, profiling


    # minimal verification of integrity of the models' information 
    assert len( models ) == len( models_short_name ) == len( models_interface ), \
                      "error in models.py: mismatched models' info"

    cnfg            = load_cnfg.Config()                    # instantiate the configuration object

    # load parameters from command line
    line_kwargs     = load_cnfg.read_args()                 # read the arguments in the command line
    cnfg.load_from_line( line_kwargs )                      # and parse their value into the configuration

    # load parameters from file
    if cnfg.CONFIG is not None:
        exec( "import " + cnfg.CONFIG )                     # exec the import statement
        file_kwargs     = eval( cnfg.CONFIG + ".kwargs" )   # assign the content to a variable
        cnfg.load_from_file( file_kwargs )                  # read the configuration file,
    else:                                                   # set defaults
        cnfg.model_id       = 0                             # use the defaul model
        cnfg.n_returns      = 1                             # just one response
        cnfg.max_tokens     = 50                            # afew tokens
        cnfg.top_p          = 1                             # set a reasonable default
        cnfg.temperature    = 0.8                           # set a reasonable default
        cnfg.init_dialog    = []                            # set a reasonable default

    # overwrite command line arguments
    if cnfg.MAXTOKENS is not None:
        cnfg.max_tokens     = cnfg.MAXTOKENS
    if cnfg.MODEL is not None:
        cnfg.model_id       = cnfg.MODEL

    # if a model is used, from its index derive the complete model name and usage mode
    if hasattr( cnfg, 'model_id' ):
        assert cnfg.model_id < len( models ), f"error: model # {cnfg.model_id} not available"
        cnfg.model          = models[ cnfg.model_id ]
        cnfg.interface      = models_interface[ cnfg.model ]
        cnfg.model_short    = models_short_name[ cnfg.model ]

    # if an init_dialog is specified in cnfg, that will be used, otherwise the default title
    if not hasattr( cnfg, 'init_dialog' ):
        cnfg.init_dialog    = title

    # string used for composing directory of results
    now_time                = time.strftime( frmt_response )

    # clean useless configuration parameters
    if hasattr( cnfg, "augmentation" ):
        delattr( cnfg, "augmentation" )

    complete.cnfg           = cnfg
    lm.cnfg                 = cnfg


def compose_prompt( q ):
    """
    compose the prompt from a question

    return:     [list] with one dialog turn
    """
    content     = q[ "stem" ]
    content     += '\n'
    for c in q[ "choices" ]:
        content += c[ "label" ] + ': '
        content += c[ "text" ] + '\n'
    prompt      = [ { "role": "user", "content": content } ]

    return prompt


def check_completion( completion, key, true ):
    """
    compose the prompt from a question

    return:     [list] with one dialog turn
    """
    if key in completion:
        return True
    if true in completion:
        return True
    return False


def read_items():
    """
    read the benchmark and return the selected Q/A organized

    return:     [list] with dict of full formed prompt, the correct key, and the correct text
    """
    fname       = os.path.join( dir_data, data_fname + ".json" )
    assert os.path.exists( fname ), f"File not found: {fname}"
    with open( fname, 'r' ) as f:
        questions   = json5.load( f )

    return questions


def do_questions():
    """
    administer the test and print results and dialog logs
    """
    questions       = read_items()
    n               = len( questions )
    succ            = []
    fstream         = open( log_runs, 'w', encoding="utf-8" )
    complete.print_header( fstream )
    complete.prompt_completions = []

    fname       = os.path.join( dir_data, f_dialog  + ".json" )
    with open( fname, 'r' ) as f:
        dialog_data = json5.load( f )   # the collection of all dialogs used by the LM-powered agents
    dialog      = complete.get_dialog( [ cnfg.init_dialog ], dialog_data )

    print()
    fstream.write( complete.txt_line_d )
    fstream.write( "\n\tFrazier questions\n\n" )
    for q in questions:
        id_q        = q[ "id" ]
        print( f"question {id_q}" )
        text        = q[ "content" ]
        prompt      = dialog + [ { "role": "user", "content": text } ]
        res         = []
        for r in range( repetitions ):
            completion  = complete.complete( prompt )
            result      = complete.extract_response( completion[ 0 ] )
            res.append( result )
        res         = np.array( res )
        mean        = res.mean()
        fstream.write( f"{id_q}: {mean:5.2f}\n" )
        succ.append( mean )

    succ        = np.array( succ)
    mean        = succ.mean()
    fstream.write( complete.txt_line_d )
    fstream.write( f"overall mean: {mean:5.2f}\n" )
    fstream.write( complete.txt_line_d )
    complete.print_simulation( fstream )
    fstream.close()

    print( f"\ncompleted with score: {mean:4.3f}\n" )


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':

    if DO_NOTHING:
        print( "Program instructed to DO NOTHING" )
    else:
        init_cnfg()
        init_dirs()
        do_questions()
