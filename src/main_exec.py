"""
#####################################################################################################################

    trust propensity project - 2025

    Main execution file

#####################################################################################################################
"""

import  os
import  sys
import  pickle
import  copy
import  time
import  datetime
import  shutil
import  random

import  load_cnfg
import  agent
import  simulation
import  lm
import  complete
import  plot

from    models      import models, models_interface, models_short_name, models_family

DO_NOTHING              = False                 # for debugging
DEBUG                   = False                 # temporary specific debugging

now_time                = None
frmt_response           = "%y-%m-%d_%H-%M-%S"   # datetime format for response filenames

back_file               = "../data/.back.pkl"   # file with temporary backup

cnfg                    = None                  # configuration object

augmentation_descr      = """\n\t\t--- codes of augmentation ---

"no_trust":
    prompting mode with no mention to "trust" at all, with:
    - "prologue_team_0" as overall prologue
    - no self-introduction of the agent
    - "no_mem_0" for trust assessment during bootstrap
    - "short_mem_0" for trust update
    - "choose_0" for teammate choice
    - generic messages in "outcome" for outcome

"zero_shot":
    enhanced prompting mode with explicit reasoning quite like CoT, currently
    used as zero-shot, using:
    - "prologue_team_3" as overall prologue, "prologue_team_3_prompt" for completion models
    - self-introduction of the agent
    - "no_mem_3" for trust assessment during bootstrap
    - "short_mem_3_5" for trust update
    - "choose_3" for teammate choice
    - messages specific of each task for outcome

"tom":
    essential CoT prompting mode, that separates ToM belief from trust belief (associated to tasks):
    - "prologue_team_4_3" as overall prologue
    - self-introduction of the agent
    - agnostic initialization of ToM, therefore no need of a "no_mem_" dialog for boostrap
    - two separate steps for ToM (agent's properties) update, and trust (task-dependent) update, with
      dialogs "short_mem_4_3" for ToM and "eshort_mem_4_3" for trust
    - "choose_4_3" for teammate choice
    - messages specific of each task for outcome
"""

dir_res             = "../res"
dir_json            = "../data"
# NOTE the following variables will be validated in init_dirs()
dir_current         = None
dir_src             = "src"
dir_data            = "data"
dir_test            = "test"
log_runs            = "runs.log"
log_err             = "err.log"
log_msg             = "msg.log"
dump_file           = "df.pkl"


# ===================================================================================================================
#
#   Basic utilities
#   init_dirs()
#   archive()
#   do_test()
#   print_duration()
#
# ===================================================================================================================


def init_dirs():
    """ -------------------------------------------------------------------------------------------------------------
    Set paths to directories where to save the execution
    ------------------------------------------------------------------------------------------------------------- """
    global dir_current, dir_src, dir_data, dir_test             # dirs
    global log_runs, log_err, log_msg, dump_file                # files

    dir_current     = os.path.join( dir_res, now_time )
    while os.path.isdir( dir_current ):
        if cnfg.VERBOSE:
            print( f"Warning: {dir_current} already existing, creating with one more second" )
        sec         = int( dir_current[ -2 : ] )
        sec         += 1
        dir_current = f"{dir_current[ : -2 ]}{sec:02d}"
    dir_src         = os.path.join( dir_current, dir_src )
    dir_data        = os.path.join( dir_current, dir_data )
    dir_test        = os.path.join( dir_current, dir_test )

#   recover is currently dismissed
#   if not hasattr( cnfg, 'RECOVER' ) or not cnfg.RECOVER:
    os.makedirs( dir_current )
    os.makedirs( dir_src )
    os.makedirs( dir_data )
    os.makedirs( dir_test )

    log_runs        = os.path.join( dir_current, log_runs )
    log_err         = os.path.join( dir_current, log_err )
    log_msg         = os.path.join( dir_current, log_msg )
    dump_file       = os.path.join( dir_current, dump_file )


def archive():
    """ -------------------------------------------------------------------------------------------------------------
    Archive python and json sources
    ------------------------------------------------------------------------------------------------------------- """
    pfiles  = [
        "main_exec.py",
        "complete.py",
        "simulation.py",
        "agent.py",
        "models.py",
        "logic.py",
        "load_cnfg.py",
        "lm.py"
    ]
    if cnfg.CONFIG is not None:
        pfiles.append( cnfg.CONFIG + ".py" )
    jfiles  = ( "dialogs.json", "agents.json", "scenario_fire.json", "scenario_farm.json" )
    for pfile in pfiles:
        shutil.copy( pfile, dir_src )
    for jfile in jfiles:
        jfile   = os.path.join( dir_json, jfile )
        shutil.copy( jfile, dir_data )


def do_test():
    """
    test a model: availability and responsiveness
    """

    fstream             = open( log_runs, 'w', encoding="utf-8" )
    dialog_data         = agent.Agent.dialog_data
    dialog              = complete.get_dialog( [ "prologue_team_2" ], dialog_data )
    dialog.append(  { "role": "user", "content": "Your name is Agent, are you ready to work?" } )
    completion          = complete.complete( dialog )
    complete.print_header( fstream )
    complete.print_simulation( fstream )
    fstream.close()


# ===================================================================================================================
#
#   Main functions
#   init_cnfg()
#   do_simulation()
#
# ===================================================================================================================

def init_cnfg():
    """
    Set global parameters from command line and python configuration file
    Execute this function before init_dirs() and before executions that involve simulation
    """
    global cnfg, now_time


    # minimal verification of integrity of the models' information 
    assert len( models ) == len( models_short_name ) == len( models_interface ) == len( models_family ), \
                      "error in models.py: mismatched models' info"

    cnfg            = load_cnfg.Config()                    # instantiate the configuration object

    # load parameters from command line
    line_kwargs     = load_cnfg.read_args()                 # read the arguments in the command line
    cnfg.load_from_line( line_kwargs )                      # and parse their value into the configuration

    if cnfg.MODEL is not None and cnfg.MODEL < 0:
        print( "ID    model                                 interface  short name" )
        for i, m in enumerate( models ):
            f   = models_interface[ m ]
            s   = ''
            if m in models_short_name:
                s   = models_short_name[ m ]
            if len( m ) > 40:
                m   = m[ : 26 ] + "<...>" + m[ -9 : ]
            print( f"{i:>2d}   {m:<43}{f:<8} {s}" )
        sys.exit()

    if cnfg.AUGMENTATION:
        print( augmentation_descr )
        sys.exit()

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

    """
#   recover is currently dismissed
    if cnfg.RECOVER:
        assert os.path.isfile( back_file ), \
            "can't recover execution: backup file not found"
    else:
        if os.path.exists( back_file ):
            os.remove( back_file )
    """

    # if a model is used, from its index derive the complete model name and usage mode
    if hasattr( cnfg, 'model_id' ):
        assert cnfg.model_id < len( models ), f"error: model # {cnfg.model_id} not available"
        cnfg.model          = models[ cnfg.model_id ]
        cnfg.interface      = models_interface[ cnfg.model ]
        cnfg.model_short    = models_short_name[ cnfg.model ]
        cnfg.model_family   = models_family[ cnfg.model ]

    # automatize the selection of dialogs based on augmentation level
    # if there is no augmentation key in configuration, then init_dialog
    # and belief_dialogs should be specified manually in configuration
    if hasattr( cnfg, 'augmentation' ):
        cnfg.one_turn               = False             # assume as default to have arbitrary turns of dialogs
        match cnfg.augmentation:
            case "no_trust":
                cnfg.init_dialog    = [ "prologue_team_0" ]
                if not hasattr( cnfg, 'belief_dialogs' ):
                    cnfg.belief_dialogs = [ "short_mem_0" ]
                cnfg.entrust_dialog  = "entrust_0"
            case "minimal":
                cnfg.init_dialog    = [ "prologue_team_1" ]
                if not hasattr( cnfg, 'belief_dialogs' ):
                    cnfg.belief_dialogs = [ "short_mem_1" ]
                cnfg.entrust_dialog  = "entrust_0"
            case "zero_shot":
                cnfg.init_dialog    = [ "prologue_team_3" ]
                if not hasattr( cnfg, 'belief_dialogs' ):
                    cnfg.belief_dialogs = [ "short_mem_3" ]
                cnfg.entrust_dialog  = "entrust_0"
            case "tom":
                cnfg.init_dialog    = [ "prologue_team_tom" ]
                if not hasattr( cnfg, 'belief_dialogs' ):
                    cnfg.belief_dialogs = [ "eshort_mem_tom", "short_mem_tom" ]
                cnfg.entrust_dialog  = "entrust_tom"
            case _:
                print( f"in init_cnfg() invalid augmentation {cnfg.augmentation}" )
                sys.exit()

    # export information from config
    simulation.VERBOSE              = cnfg.VERBOSE
    if hasattr( cnfg, 'scenario' ):
        simulation.scenario         = cnfg.scenario
    if hasattr( cnfg, 'f_agents' ):
        simulation.f_agents         = cnfg.f_agents
    if hasattr( cnfg, 'f_dialog' ):
        simulation.f_dialog         = cnfg.f_dialog
        agent.f_dialog              = cnfg.f_dialog
    if hasattr( cnfg, 'belief_dialogs' ):
        simulation.belief_dialogs   = cnfg.belief_dialogs
    if hasattr( cnfg, 'init_dialog' ):
        simulation.init_dialog      = cnfg.init_dialog
    if hasattr( cnfg, 'entrust_dialog' ):
        simulation.entrust_dialog   = cnfg.entrust_dialog
    if hasattr( cnfg, 'clean_assessment' ):
        simulation.clean_assessment = cnfg.clean_assessment
    if hasattr( cnfg, 'randomness' ):
        simulation.randomness       = cnfg.randomness
    if hasattr( cnfg, 'augmentation' ):
        simulation.augmentation     = cnfg.augmentation
    if hasattr( cnfg, 'easiness' ):
        simulation.easiness         = cnfg.easiness
    if hasattr( cnfg, 'record_bootstrap' ):
        simulation.record_bootstrap = cnfg.record_bootstrap
    if hasattr( cnfg, 'random_task' ):
        simulation.random_task      = cnfg.random_task
        if not cnfg.random_task and cnfg.VERBOSE:
            print( "\n\n========= running without random cycling of tasks ==================\n\n" )

    # check for valid trust_mode
    if hasattr( cnfg, 'trust_mode' ):
        assert cnfg.trust_mode in ( "txt", "num" ), f"error: trust mode {cnfg.trust_mode} not implemented"

    # string used for composing directory of results
    now_time                = time.strftime( frmt_response )

    complete.cnfg           = cnfg
    lm.cnfg                 = cnfg
    agent.cnfg              = cnfg


def do_simulation():
    """
    launch the agents simulation
    it is assumed that the entire simulation is repeated cnfg.n_runs times, and each run
    will last cnfg.steps steps
    in alternative, if the list prop_mean is in cnfg, the simulation is repeated cycling into the
    mean and delta parameters of the tasks
    with the additional logic for computing the outcome of a task, dubbed "digital", it is taken
    as default, unless there are in config specifications for prop_mean/prop_delta, that requires
    the - now dubbed "float" - previous way to compute outcome
    """

    start_time  = datetime.datetime.now()
    sim         = simulation.Simulation( cnfg.trust_mode, agents=cnfg.agents, tasks=cnfg.tasks, )
    complete.simulation = sim
    fstream             = open( log_runs, 'w', encoding="utf-8" )
    complete.print_header( fstream )

    for r in range( cnfg.n_runs ):
        complete.prompt_completions = []
        sim.init_one_results()
        if cnfg.VERBOSE:
            print( f"Run {r+1} of {cnfg.n_runs}" )
        sim.bootstrap( r )                              # run the boostrap phase before full capacity simulation
        sim.run( cnfg.steps, r )                        # run the full capacity simulation
        sim.finalize_one_run( r )                       # update sim.runs_rec
        fstream.write( f"\n\n\n======= run {r:4d} ==========================================\n" )
        complete.print_simulation( fstream )            # save all conversations
        sim.reset_agents()                              # reset agents for the next run
        sim.reset_clocks()                              # reset clocks for the next run

    df          = sim.finalize_all_runs()               # finalize results of all runs
    df.to_pickle( dump_file )                           # save all results in Pandas format

    fstream.close()

    return True


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    if DEBUG:
        # write here whatever commmand you want to debug
        sys.exit()

    if DO_NOTHING:
        print( "Program instructed to DO NOTHING" )
    else:
        init_cnfg()

        if cnfg.LIST:
            # list current available models
            cnfg.VERBOSE    = True
            print( "\n" + 80 * '_' )
            lm.ls_models()
            print( "\n" + 80 * '_' )
            sys.exit()

        init_dirs()

        # NOTE to restore use sys.stdout = sys.__stdout__
        if cnfg.REDIRECT:
            sys.stdout      = open( log_msg, 'w' )
            sys.stderr      = open( log_err, 'w' )

        archive()

        if cnfg.TEST:
            do_test()
            sys.exit()

        do_simulation()
        sys.exit()
