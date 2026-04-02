"""
#####################################################################################################################

    HURST project - 2024

    Simulation management

#####################################################################################################################
"""

import  os
import  sys
import  datetime
import  json5
import  json                                            # necessary, see NOTE in step_training()
import  random
import  torch
import  gc

import  numpy    as np
import  pandas   as pd

import  complete
import  agent

belief_dialogs          = None                          # list of dialog titles for updating trust belief
init_dialog             = None                          # list of initial dialgoues
entrust_dialog          = None                          # the dialog for asking the model to choose a partner
randomness              = 0.02                          # randomnes level in task outcome
easiness                = 1                             # global level of task outcome easiness
augmentation            = "minimal"                     # level of augmentation in all dialog messages
token_increase          = 3                             # ratio of token increase for some augmentation
clean_assessment        = True                          # clean the trust assessment completion
random_task             = True                          # randomize the choice of tasks during perform_task()
record_bootstrap        = False                         # do record the bootstrap phase too

data_dir                = "../data"                     # directory with all input data
scenario                = "scenario_fire"               # filename with the tasks of a scenario
f_dialog                = "dialogs"                     # filename with dialog turns
f_agents                = "agents"                      # filename with the agents in use

VERBOSE                 = False                         # will be overwritten by cnfg
TRAIN                   = False                         # will be overwritten by cnfg


# ===================================================================================================================
#
#   Simulation class
#   methods:
#       init_tasks()
#       init_outcomes()
#       init_trustor()
#       init_agents()
#       init_belief_gen()
#       init_belief_task()
#       reset_agents()
#       init_one_results()
#       init_runs_results()
#       log_results()
#       compose_task()
#       agent_outcome_float()
#       agent_outcome()
#       bootstrap()
#       run()
#       step_clocks()
#       reset_clocks()
#       get_clock()
#       increase_max_tokens()
#       reset_max_tokens()
#       perform_task()
#       finalize_one_run()
#       finalize_all_runs()
# ===================================================================================================================

class Simulation():
    """
    manage the overall simulation
    """
    records_columns = (
        "clock",
        "task",
        "trustee",
        "decision",
        "outcome",
        "success",
        "failure",
    )

    def __init__( self, trust_mode, agents=None, tasks=None ):
        """
        initialize the simulation
        args:
            agents      [list] of names of the agents, if None all will be used
            tasks       [list] of names of tasks, if None all will be used
            trust_mode  [str] the code for how trust is managed by the LM
        """
        start_time          = datetime.datetime.now()
        self.trust_mode     = trust_mode
        self.agents         = []                # list of agents in the simulation
        self.agent_names    = []                # list of the names of agents
        self.tasks          = []                # list of tasks in the simulation (as in json file)
        self.task_names     = []                # list of the names of tasks
        self.current_task   = {}                # storage for the current task messages, with filled slots
        self.outcomes       = {}                # dict of outcomes in the simulation (as in json file)
        self.idx_task       = -1                # index of the current task
        self.idx_agent      = -1                # index of the current agent
        self.idx_partner    = -1                # index of the current trustee
        self.records        = {}                # record of all tasks results and infos for one run
        self.runs_rec       = {}                # record over multiple runs
        self.boostrap_stage = False             # flags if bootstrap is in execution

        fname           = os.path.join( data_dir, f_dialog + ".json" )
        assert os.path.exists( fname ), f"File not found: {fname}"
        with open( fname, 'r' ) as f:
            ddata = json5.load( f )

        fname           = os.path.join( data_dir, scenario + ".json" )
        assert os.path.exists( fname ), f"File not found: {fname}"
        with open( fname, 'r' ) as f:
            tdata = json5.load( f )

        self.init_tasks( tasks, tdata )
        self.init_outcomes( ddata )
        self.init_runs_results()

        fname           = os.path.join( data_dir, f_agents + ".json" )
        assert os.path.exists( fname ), f"File not found: {fname}"
        with open( fname, 'r' ) as f:
            data = json5.load( f )
        self.init_agents( agents, data )
        self.init_trustor()
        self.max_tokens = complete.cnfg.max_tokens  # save the default max_tokens, that could be increased
        self.clock      = 0


    def init_tasks( self, tasks, data ):
        """
        initialize the tasks
        self.tasks is a list of dictionary with all taks information, self.task_names is the corresponding
        list with the names of the tasks

        args:
            tasks       [list] of names of the tasks or None to take all
            data        [list] of content in the json file of entities
        """

        verb            = 3                         # the most verbose description in the scenarios
        task_messages   = True                      # use additional messages for describing a task and its outcome
        if augmentation == "no_trust":
            verb = 1                                # minimal description
            task_messages   = False
        if augmentation == "minimal":
            verb = 1                                # minimal description
            task_messages   = False
        if augmentation == "medium":
            verb = 2                                # medium description
            task_messages   = False
        descr_key       = f"descr_{verb}"           # get the proper "descr_X" according to augmentation

        for d in data:                              # get all tasks in the json file
            if d[ "id" ] == "tasks":
                all_tasks   = d[ "content" ]

        for t in all_tasks:
            if tasks is not None:                   # if there is a list of tasks, take only those
                if t[ "name" ] not in tasks:
                    continue
            t_dict  = {
                    "description":  t[ descr_key ],
                    "vars" :        t[ "vars" ],
                    "prop" :        t[ "prop" ]
            }
            if task_messages:                       # additional messages for describing a task and its outcome
                t_dict[ "o_succ" ] =    t[ "o_succ" ]
                t_dict[ "o_comp" ] =    t[ "o_comp" ]
                t_dict[ "o_reli" ] =    t[ "o_reli" ]
                t_dict[ "o_will" ] =    t[ "o_will" ]
            self.tasks.append( t_dict )
            self.task_names.append( t[ "name" ] )


    def init_outcomes( self, data ):
        """
        initialize the generic messages to give depending on the outcomes, from json file
        it is exepcted that the entry "outcome" in the json file has text for the following keys:
                intro, success, f_comp, f_reli, f_will
        NOTE: these messages are used only when a task has no specific outcome messages
        args:
            data        [list] of content in the json file of dialogs
        """
        for d in data:
            if d[ "id" ] == "outcome":
                all_outs    = d[ "content" ]
        for o in all_outs:
            self.outcomes[ o ]  = all_outs[ o ][ "content" ]


    def init_trustor( self ):
        """
        initialize the trustor agent
        """
        no_intro        = False
        no_ass          = False
        if augmentation == "no_trust":
            no_intro        = True                          # avoid intrucing itself in the agent preamble
            no_ass          = True                          # do not include the assistant role in the preamble
        if augmentation == "minimal":
            no_intro        = True                          # avoid intrucing itself in the agent preamble
            no_ass          = True                          # do not include the assistant role in the preamble

        self.trustor                = agent.Trustor()
        self.trustor.trust_mode     = self.trust_mode       # how trust is managed ("txt", "num")
        self.trustor.belief_dialog  = belief_dialogs        # title of dialogs for trust belief update
        if len( init_dialog ):
            self.trustor.set_preamble( init_dialog, no_intro=no_intro, no_ass=no_ass )



    def init_agents( self, agents, data ):
        """
        initialize the agents
        args:
            agents      [list] of names of the agents or None to take them all
            data        [list] of content in the json file of entities
        """
        for d in data:
            if d[ "id" ] == "agents":
                all_agents  = d[ "content" ]
        for a in all_agents:
            name        = a[ "name" ]
            if agents is not None:                          # if there is a list of tasks, take only those
                if name not in agents:
                    continue
            new_agent       = agent.Trustee( name, properties=a[ "prop" ] )
            self.agents.append( new_agent )
            self.agent_names.append( name )


    def init_belief_gen( self ):
        """
        initialize the ToM inference of the agents
        """
        for a in self.agents:
            self.trustor.init_belief_gen( a )


    def init_belief_task( self ):
        """
        initialize the trust belief of the agents
        """
        for t in self.task_names:
            for a in self.agents:
                self.trustor.init_belief_task( a, t )


    def reset_agents( self ):
        """
        reset the memory of the agent
        """
        self.trustor.clear_memories()


    def init_one_results( self ):
        """
        initialize the structure of result over one single simulation run
        NOTE that this function is typically executed outside, in the loop
        over multiple runs
        """
        for c in self.records_columns:
            self.records[ c ]       = []


    def init_runs_results( self ):
        """
        initialize the structure for results over multiple simulation runs
        the columns are the same of init_one_results with some additional info
        """
        for c in self.records_columns:
            self.runs_rec[ c ]      = []
        self.runs_rec[ "run" ]      = []
        self.runs_rec[ "model" ]    = []
        self.runs_rec[ "family" ]   = []
        self.runs_rec[ "scen" ]     = []
        self.runs_rec[ "augm" ]     = []


    def log_results( self, outcome, decision, success ):
        """
        log current results for one step of simulation

        args:
            outcome         [float] outcome ot the task
            decision        [bool] True if the trustor entrusted the trustee for the task
            success         [bool] True if the trustor entrusted the trustee, and the task succeeded

        """
        task            = self.tasks[ self.idx_task ]       # the task object
        task_name       = self.task_names[ self.idx_task ]  # the task name
        trustee         = self.agents[ self.idx_agent ]     # the trustee object in charge for the task

        failure         = decision and not success

        self.records[ "clock" ].append( self.clock )
        self.records[ "task" ].append( task_name )
        self.records[ "trustee" ].append( trustee.name )
        self.records[ "decision" ].append( decision )
        self.records[ "outcome" ].append( outcome )
        self.records[ "success" ].append( success )
        self.records[ "failure" ].append( failure )


    def compose_task( self, i_task ):
        """
        compose the complete description of a task, and validate the other messages filling
        the proper slots

        args:
            i_task      [int] index of the task to issue
        return:
            [str] the complete task description
        """
        task        = self.tasks[ i_task ]
        descr       = task[ "description" ]
        var_dict    = task[ "vars" ]
        if not len( var_dict ):
            return descr
        d_values    = dict()
        for v in var_dict:
            d_values[ v ]   = random.choice( var_dict[ v ] )

        task_desc   = complete.fill_placeholders( descr, d_values )

        # keep the name slot, for later filling with the partner's name, yet unknown
        d_values[ "name" ]  = "{name}"
        # while fill the other slots, if detailed descriptions are available
        if "o_succ" in task.keys():
            descr                           = task[ "o_succ" ]
            self.current_task[ "o_succ" ]   = complete.fill_placeholders( descr, d_values )
        if "o_comp" in task.keys():
            descr                           = task[ "o_comp" ]
            self.current_task[ "o_comp" ]   = complete.fill_placeholders( descr, d_values )
        if "o_reli" in task.keys():
            descr                           = task[ "o_reli" ]
            self.current_task[ "o_reli" ]   = complete.fill_placeholders( descr, d_values )
        if "o_will" in task.keys():
            descr                           = task[ "o_will" ]
            self.current_task[ "o_will" ]   = complete.fill_placeholders( descr, d_values )

        return task_desc


    def agent_outcome_float( self, task_props, agent_props ):
        """
        compute the outcome of a collaborating agent, using integer properties {0,1,2} for
        agent, and {1,2,4} for taks, the scheme of the dot product has been updated several time, the main one are
        kept below commented
                task		agent	product	outcome
                1 2 4		0 1 2	10	1 - r
                1 2 4		1 0 2	 9	1 - 2 * r   if e>-1 else 5 * r
                1 2 4		0 2 1	 8	1 - 3 * r   if e>0  else 4 * r
                1 2 4		2 0 1	 6	3 * r       if e<2  else 1 - 4 * r
                1 2 4		1 2 0	 5	2 * r
                1 2 4		2 1 0	 4	r
        in this computation easiness can take values -1, 0, 1, 2

        even if the outcome is now a global computation, and not directly depends on single properties match,
        there is an additional logic for producing, in case of unsuccess, the most appropriate
        textual description of the failure

        args:
            task_props  [tuple] properties of the task
            agent_props [tuple] properties of the agent
        return:
            [tuple] ( [bool], [int], [float] ) if succeded, failure code, numerical outcome
        """
        task_vec    = np.array( task_props )
        agent_vec   = np.array( agent_props )
        task_i1     = task_vec.argsort()[ 1 ]           # the index of the value "1"
        task_i2     = task_vec.argsort()[ -1 ]          # the index of the value "2"
        outcome     = np.dot( task_vec, agent_vec )
        outcome     = min( 10, outcome )                # keep the dot product within the allowed range
 
        # keep the index of the property responsible for failure,
        # with a random inizialization, for cases where the reason of failure are ambiguous
        failure     = random.choice( [ task_i1, task_i2 ] )

        # use aliases to write less...
        r   = randomness

        match easiness:
            case -1:
                if outcome > 9:
                    p           = 1 - r
                elif outcome > 7:
                    p           = ( outcome - 4 ) * r
                else:
                    p           = ( outcome - 3 ) * r
            case 0:
                if outcome > 8:
                    p           = 1 - ( 11 - outcome ) * r
                elif outcome > 7:
                    p           = ( outcome - 4 ) * r
                else:
                    p           = ( outcome - 3 ) * r
            case 1:
                if outcome > 7:
                    p           = 1 - ( 11 - outcome ) * r
                else:
                    p           = ( outcome - 3 ) * r
            case 2:
                if outcome > 7:
                    p           = 1 - ( 11 - outcome ) * r
                elif outcome > 5:
                    p           = 1 - ( 10 - outcome ) * r
                else:
                    p           = ( outcome - 3 ) * r

        match round( outcome ):
            case 9:
                failure     = task_i1                       # the highest task score has met, blame the medium one
            case 8:
                failure     = task_i2                       # the highest task score has not met
            case 6:
                failure     = task_i1                       # the highest task score has met, blame the medium one
            case 5:
                failure     = task_i2                       # the highest task score has not met
            case 4:
                failure     = task_i2                       # the highest task score has not met
            case _:
                pass

        succ    = p > random.random()

        return succ, outcome, failure


    def agent_outcome( self, task, task_desc ):
        """
        compute the outcome of a collaborating agent
        the possibilities of failure are evaluated against the internal properties of the agent in combination
        with the properties of the task at hand
        there is also the possibility to mix a responsibility of the main agent in the success/failure, with
        the main_role parameter, but it cannot be used for the digital outcome computation

        args:
            task        [dict] the task structure
            task_desc    [str] the description of the task
            main_role   [float] the fraction of responsibility  of the main agent
        return:
            [tuple] ( [bool], [bool], [str] ) if succeded, if correct choice, description of the outcome
        """
        tcompetence     = task[ "prop" ][ 'competence' ]
        treliability    = task[ "prop" ][ 'reliability' ]
        twillingness    = task[ "prop" ][ 'willingness' ]

        agent           = self.agents[ self.idx_agent ]
        name            = agent.name
        competence      = agent.prop[ 'competence' ]
        reliability     = agent.prop[ 'reliability' ]
        willingness     = agent.prop[ 'willingness' ]

        task_props      = tcompetence, treliability, twillingness
        agent_props     = competence, reliability, willingness

        succ, out, fail = self.agent_outcome_float( task_props, agent_props )
        text            = self.outcomes[ "intro" ].format( name=name, task=task_desc ) + '\n'

        if succ:                                        # success
            if "o_succ" in task.keys():
                text    += self.current_task[ "o_succ" ].format( name=name )
            else:
                text    += self.outcomes[ "success" ].format( name=name )
            return True, out, text

        match fail:                                  # failure is the index to the task property more responsible
            case 0:
                if "o_comp" in task.keys():
                    text    += self.current_task[ "o_comp" ].format( name=name )
                else:
                    text    += self.outcomes[ "f_comp" ].format( name=name )
                return False, out, text

            case 1:
                if "o_reli" in task.keys():
                    text    += self.current_task[ "o_reli" ].format( name=name )
                else:
                    text    += self.outcomes[ "f_reli" ].format( name=name )
                return False, out, text

            case 2:
                if "o_will" in task.keys():
                    text    += self.current_task[ "o_will" ].format( name=name )
                else:
                    text    += self.outcomes[ "f_will" ].format( name=name )
                return False, out, text


    def bootstrap( self, current_run ):
        """
        run a preliminary turn of tasks in order to bootstrap a trust belief in every agent towards all the others

        args:
            current_run [int]   the current run, just to display
        """
        self.boostrap_stage     = True

        self.init_belief_gen()
        if augmentation == "tom":
            self.init_belief_task()

        if VERBOSE:
            print( f"\trun {current_run+1:3d}\t-- doing bootstrap" )
        for i_a in range( len( self.agents ) ):
            for i_t in range( len( self.tasks ) ):
                self.perform_task( i_agent=i_a, i_task=i_t )

        self.boostrap_stage     = False


    def run( self, steps, current_run ):
        """
        run the simulation for a given number of steps
        at every step cycles into the existing agents, and if the previous perform_task
        has succeeded (the agent has been entrusted), steps into a new task, otherwise
        continue to propose new agents for the same task
        NOTE: if no one of the agents is trusted, the model continue to loop, this is counted as
        failure in infstat.py by eval_success()

        args:
            steps       [int]   steps to run
            current_run [int]   the current run, just to display
        """
        i_a     = 0
        trusted = True
        for s in range( steps ):
            if VERBOSE:
                print( f"\trun {current_run+1:3d}\tstep {s+1:4d} of {steps}" )
            if trusted:
                if random_task:
                    i_task  = random.choice( range( len( self.tasks ) ) )
                else:
                    i_task  = ( self.idx_task + 1 ) % len( self.tasks )
            trusted     = self.perform_task( i_agent=i_a, i_task=i_task )
            i_a = ( i_a + 1 ) % len( self.agents )


    def step_clocks( self ):
        """
        step the clock one unit for the simulation and for all agents
        """
        self.clock      += 1
        self.trustor.step_clock()


    def reset_clocks( self ):
        """
        reset the clock for the simulation and for all agents
        """
        self.clock      = 0
        self.trustor.reset_clock()


    def get_clock( self ):
        """
        get the current clock
        """
        return self.clock


    def increase_max_tokens( self ):
        """
        increase max_tokens completion, depending on augmentation
        """
        if augmentation in [ "no_trust", "minimal", "medium", "medium_task", "fine_tuning", "zero_shot" ]:
            return False
        self.max_tokens             = complete.cnfg.max_tokens  # save the default max_tokens
        complete.cnfg.max_tokens    = int( token_increase * self.max_tokens )
        return True


    def reset_max_tokens( self ):
        """
        reset max_tokens to the configured value
        """
        complete.cnfg.max_tokens    = self.max_tokens


    def perform_task( self, i_task=None, i_agent=None, random_task=True ):
        """
        attempt to assign a task to an agent, and if accepted, manage the task outcome
        if no agent is specified, one is selected, same for task

        args:
            i_task      [int] index of the task to issue
            i_agent     [int] index of the trustee to assign the taks
            random_task [bool] randomize the choise of a task
        """
        if not self.boostrap_stage or record_bootstrap:
            self.step_clocks()                              # advance one time step in the simulation
        if i_task is None:
            if random_task:
                i_task  = random.choice( range( len( self.tasks ) ) )
            else:
                i_task  = ( self.idx_task + 1 ) % len( self.tasks )
        if i_agent is None:
            i_agent  = ( self.idx_agent + 1 ) % len( self.agents )
        self.idx_task   = i_task
        self.idx_agent  = i_agent
        task            = self.tasks[ self.idx_task ]       # the task object to handle
        task_name       = self.task_names[ self.idx_task ]  # the name of the task
        task_desc       = self.compose_task( self.idx_task )# the description of the task
        trustee         = self.agents[ self.idx_agent ]
        # NOTE that the result on the task is computed even before the decision, for using it
        # if in bootstrap phase
        task_result     = self.agent_outcome( task, task_desc )    # ( success, outcome, description )
        outcome         = task_result[ 1 ]

        # there is no need of taking a decision when in bootstrap phase
        if not self.boostrap_stage:
            # get the prompt for entrusting the trustee, using the appropriate key in the json file of messages
            if augmentation == "tom":
                prompt      = self.trustor.prompt_tom_entrust(
                        trustee,
                        task_desc,
                        task_name,
                        entrust_dialog=entrust_dialog
                )
            else:                                           # use the standard general trust belief
                prompt      = self.trustor.prompt_entrust( trustee, task_desc, entrust_dialog=entrust_dialog )
            completion      = complete.complete( prompt )   # and its completion
            # the completion is an arbitrary text, from which yes/no about entrusting should be extracted
            decision        = complete.extract_decision( completion[ 0 ] )

            # the trustor decided NOT to entrust the trustee, therefore its behavior on the task is not used for
            # updating the trust belief, but just for deriving the outcome and store it in the results log
            # unless in the boostrap stage, where it is necessary to fill trust beliefs according to outcome
            if not decision:
                self.log_results( outcome, False, False )
                return False

        res_descr       = task_result[ -1 ]
        success         = task_result[ 0 ]
        prompt          = self.trustor.prompt_belief_gen( trustee, res_descr, task_desc )
        self.increase_max_tokens()                          # use more tokens, depending on augmentation
        completion      = complete.complete( prompt )       # call the language model and gather completion
        self.reset_max_tokens()                             # revert to normal number of token
        if clean_assessment:
            # do not store the raw completion of the descriptive trust, do some cleaning before
            trust_t         = complete.clean_assessment( completion[ 0 ] )
        else:
            trust_t         = completion[ 0 ]

        if trust_t is not None:                                 # do not update the belief when invalid completion
            if "txt" in self.trust_mode:
                trust       = [ -1, trust_t ]                   # when numeric trust is ignored store a -1
            if self.trust_mode == "num":
                trust_n     = complete.extract_trust( completion[ 0 ] )
                trust       = [ trust_n, "" ]                   # when descriptive trust is ignored store a ""
            self.trustor.store_belief_gen( trustee, trust )     # store the trust (or ToM) in general trust belief

        # if the task specific trust belief is in use manage the task-specific trust belief,
        if augmentation == "tom":
            prompt          = self.trustor.prompt_belief_task( trustee, res_descr, task_desc, task_name )
            self.increase_max_tokens()
            completion      = complete.complete( prompt )
            self.reset_max_tokens()
            if clean_assessment:
                # do not store the raw completion of the descriptive trust, do some cleaning before
                trust_t         = complete.clean_assessment( completion[ 0 ] )
            else:
                trust_t         = completion[ 0 ]

            if trust_t is not None:                             # do not update the belief when invalid completion
                if "txt" in self.trust_mode:
                    trust       = [ -1, trust_t ]
                if self.trust_mode == "num":
                    trust_n     = complete.extract_trust( trust_t )
                    trust       = [ trust_n, "" ]
                self.trustor.store_belief_task( trustee, task_name, trust )

        if not self.boostrap_stage or record_bootstrap:
            self.log_results( outcome, True, success )

        return True


    def finalize_one_run( self, current_run ):
        """
        cumulate results of one simulation run inside the multirun results record
            current_run [int]   the current run

        """
        for k in self.records.keys():
            self.runs_rec[ k ]  += self.records[ k ]
        n                       = len( self.records[ "clock" ] )
        self.runs_rec[ "run" ]  += n * [ current_run ]


    def finalize_all_runs( self ):
        """
        cumulate results of all simulation runs
        returns:
            [[pandas.core.frame.DataFrame]]
        """
        n                           = len( self.runs_rec[ "clock" ] )
        self.runs_rec[ "model" ]    = n * [ complete.cnfg.model_short ]
        self.runs_rec[ "family" ]   = n * [ complete.cnfg.model_family ]
        scen                        = scenario.replace( "scenario_", "" )
        self.runs_rec[ "scen" ]     = n * [ scen ]
        self.runs_rec[ "augm" ]     = n * [ augmentation ]

        d           = dict()
        for k in self.runs_rec.keys():
            d[ k ]  = np.array( self.runs_rec[ k ] )
        df          = pd.DataFrame( d )

        return df
