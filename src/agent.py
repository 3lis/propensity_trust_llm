"""
#####################################################################################################################

    HURST project - 2024

    Agents management

#####################################################################################################################
"""

import  os
import  json5
import  copy
import  numpy       as np
import  random
import  complete
import  logic

cnfg                    = None                          # configuration object

data_dir                = "../data"                     # directory with all input data
f_dialog                = "dialogs.json"                # filename with dialog turns


# ===================================================================================================================
#
#   Trustee class
#   methods:
#
# ===================================================================================================================

class Trustee():
    """
    the base agent

    there are N possible types of properties held by the agent, current default is N=3,
    each property has a numerical value such that the sum of all values should be N, and a text description
    """
    default_prop    = {
        'competence'    : 1.0,
        'reliability'   : 1.0,
        'willingness'   : 1.0
    }

    def __init__( self, name, properties=None ):
        """
        initialize an agent
        args:
            name        [str] name of the agent
            properties  [dict] of properties, with as value a list with number and text
        """
        self.name       = name
        self.prop       = dict()
        self.prop.update( self.default_prop )
        if properties is not None:
            self.prop.update( properties )



# ===================================================================================================================
#
#   Trustor class
#   methods:
#       store_long_memory()
#       print_long_memory()
#       store_belief_gen()
#       init_belief_gen()
#       store_belief_task()
#       check_belief_task()
#       get_long_memory()
#       get_belief_gen_log()
#       clear_memories()
#       step_clock()
#       get_clock()
#       reset_clock()
#       introduce()
#       set_preamble()
#       add_preamble()
#       prompt_belief_gen()
#       prompt_belief_task()
#       prompt_entrust()
#       prompt_tom_choose_trustee()
#
# ===================================================================================================================

class Trustor():
    """
    the main agent

    trust can be evaluated and used in different modes, according to trust_mode, with current
    possible values:
        "txt":      only the descriptive assessments of trust are asked and used
        "num":      only the numerical assessments of trust are asked and used
        NOTE: "num" mode has been discontinued, therefore is not updated with all augmentation methods
    """
    trust_mode      = "txt"
    belief_dialog   = "belief_gen_2"    # titles of the dialog to use for trust belief updates
                                        # there is just one title for generic trust, two titles for ToM augmentation:
                                        # [ trust_gen_title, trust_task_title ]

    default_tom     = "I must wait to see {name} at work, to infer what his aptitudes are in terms of competence, reliability, and willingness"
    default_trust   = "I must wait to see {name} at work, to be able to trust their ability to accomplish this task"

    fname       = os.path.join( data_dir, f_dialog )
    with open( fname, 'r' ) as f:
        dialog_data = json5.load( f )   # the collection of all dialogs used by the LM-powered agents


    def __init__( self, name="Harmony" ):
        """
        initialize an agent
        args:
            name        [str] name of the agent
        """
        self.name           = name
        self.preamble       = []        # chat that is the preamble for all interactions with a LM
        self.belief_gen     = {}        # general believes: other agent's names as key and current trust as value,
        self.belief_task    = {}        # task-specific trust believes with nested keys for agents and tasks
        self.clock          = 0
        self.preamble       = []        # chat that is the preamble for all interactions with a LM


    def store_belief_gen( self, trustee, trust ):
        """
        update the general trust belief for a trustee
        args:
            trustee     [Trustee] the collaborating agent
            trust       [list] [ [int] [str] ] numeric and textual trust judgment
        """
        self.belief_gen[ trustee.name ] = trust


    def init_belief_gen( self, trustee ):
        """
        initialize the general trust belief with the default belief
        args:
            trustee    [Agent] the collaborating agent
        """
        name                    = trustee.name
        tom                     = self.default_tom.format( name=name )
        self.belief_gen[ name ]  = [ -1, tom ]


    def store_belief_task( self, trustee, task_name, trust ):
        """
        update the task specific trust belief
        args:
            trustee    [Agent] the collaborating agent
            task_name   [str] the name of the task
            trust       [list] [ [int] [str] ] numeric and textual trust judgment
        """
        a       = trustee.name
        if a in self.belief_task.keys():
            self.belief_task[ a ][ task_name ]   = trust
        else:
            self.belief_task[ a ] = { task_name: trust }


    def init_belief_task( self, trustee, task_name ):
        """
        initialize the general trust belief with the default belief
        args:
            trustee    [Agent] the collaborating agent
        """
        name    = trustee.name
        trust   = self.default_trust.format( name=name )
        trust   = [ -1, trust ]
        self.store_belief_task( trustee, task_name, trust )


    def check_belief_task( self, trustee, task_name ):
        """
        verify the presence of agent/task in the task specific trust belief
        args:
            trustee    [Agent] the collaborating agent
            task_name   [str] the name of the task
        """
        a       = trustee.name
        if not a in self.belief_task.keys():
            return False
        if task_name in self.belief_task[ a ].keys():
            return True
        return False


    def clear_memories( self ):
        """
        clear all memories
        """
        self.belief_gen      = {}
        self.belief_task     = {}


    def step_clock( self ):
        """
        step the clock one unit
        """
        self.clock      += 1


    def get_clock( self ):
        """
        get the current clock
        returns:
            [int] clock value
        """
        return self.clock


    def reset_clock( self ):
        """
        reset the current clock
        """
        self.clock      = 0


    def introduce( self, no_ass ):
        """
        introduce the agent herself in the context of a dialog
        args:
            no_ass  [bool] do not include the assistant role
        return:
            [list]  the conversation introducing the agent
        """
        prompt      = []
        prompt.append( { "role": "user", "content": f"Your name is {self.name}" } )
        if no_ass:
            return prompt
        prompt.append( { "role": "assistant", "content": f"I'm {self.name}, got it." } )
        return prompt


    def set_preamble( self, init_dialog, no_intro=False, no_ass=False ):
        """
        compose the prompt preamble that is fixed for all furhter LM interactions
        args:
            init_dialog [list] of preliminary dialogues
            no_intro    [bool] do not include the self-introduction
            no_ass      [bool] do not include the assistant role
        return:
            [list]  the conversation preamble
        """
        if no_intro:
            prompt      = complete.get_dialog( init_dialog, self.dialog_data )
        else:
            intro       = self.introduce( no_ass=no_ass )
            prompt      = complete.get_dialog( init_dialog, self.dialog_data, intro=intro )
        self.preamble   = prompt


    def add_preamble( self, prompt ):
        """
        prefix the preamble to a prompt
        NOTE: it is necessary to make a deep copy of the preamble to prevent unintended modifications
        args:
            prompt  [list]  the conversation prompt or [str] for complete mode models
        return:
            [list]  the complete conversation or [str] for complete mode models
        """
        if isinstance( self.preamble, str ):
            if isinstance( prompt, str ):
                return self.preamble + prompt
            if isinstance( prompt[ 0 ], dict ):
                return self.preamble + prompt[ 0 ][ "content" ]

        pre             = copy.deepcopy( self.preamble )
        prompt          = pre + prompt
        return prompt


    def prompt_belief_gen( self, trustee, outcome, task_desc ):
        """
        get the prompts for updating the general trust belief
        the prompt can be composed by a chain of propmt+completions, to perform CoT strategies

        args:
            trustee    [Agent] the collaborating agent
            outcome     [str] description of the outcome of the collaboration
            task_desc    [str] description of the current task
        return:
            [list]      the prompt dialog
        """
        name        = trustee.name
        var         = { "outcome" : outcome, "name": name, "task": task_desc }

        title       = self.belief_dialog[ 0 ]

        num, text   = self.belief_gen[ name ]
        if "txt" in self.trust_mode:
            var[ "text" ]   = text
        if self.trust_mode == "num":
            var[ "num" ]   = num
        dialog  = complete.get_belief_dialog(
                title,
                self.dialog_data,
                trust_mode  = self.trust_mode,
                var         = var
        )
        prompt      = self.add_preamble( dialog )
        prompt      = complete.collapse_roles( prompt )

        return prompt


    def prompt_belief_task( self, trustee, outcome, task_desc, task_name ):
        """
        get the prompts for updating the extended general trust belief
        the prompt can be composed by a chain of propmt+completions, to perform CoT strategies

        args:
            trustee    [Agent] the collaborating agent
            outcome     [str] description of the outcome of the collaboration
            task_desc   [str] description of the current task
            task_name   [str] name of the task
        return:
            [list]      the prompt dialog
        """
        name        = trustee.name
        var         = { "outcome" : outcome, "name": name }

        # the following are just emergency actions, missing agent/task should be handled by the calling function
        if not name in self.belief_task.keys():
            if cnfg.VERBOSE:
                print( "\nErrror in prompt_belief_task: no memory entry for agent {name}" )
            return None
        if not task_name in self.belief_task[ name ].keys():
            if cnfg.VERBOSE:
                print( "\nErrror in prompt_belief_task: no memory entry for task {task_name}" )
            return None

        title       = self.belief_dialog[ -1 ]

        num, text   = self.belief_task[ name ][ task_name ]
        if "txt" in self.trust_mode:
            var[ "text" ]   = text
        if self.trust_mode == "num":
            var[ "num" ]   = num
        dialog  = complete.get_belief_dialog(
                title,
                self.dialog_data,
                trust_mode  = self.trust_mode,
                var         = var
        )

        prompt      = complete.collapse_roles( dialog )

        return prompt



    def prompt_entrust( self, trustee, task_desc, entrust_dialog="choose_2" ):
        """
        get the prompt for choosing a trustee

        args:
            trustee         [Trustee] the agent to entrust
            task_desc       [str] description of the task
            entrust_dialog  [str] key in the json file of dialogs, for the appropriate choosing trustee sentence
        return:
            [str] the prompt
        """
        found       = False
        for d in self.dialog_data:
            if d[ 'id' ] == entrust_dialog:
                d_str   = d[ 'content' ]
                found   = True
                break

        if not found:
            if cnfg.VERBOSE:
                print( f"\nErrror in prompt_entrust: {entrust_dialog} not found" )
            return None

        name        = trustee.name
        prmpt       = ''
        if "txt" in self.trust_mode:
            prmpt   += d_str[ 'intro_txt' ].format( task=task_desc )
        if self.trust_mode == "num":
            prmpt   += d_str[ 'intro_num' ].format( task=task_desc )

        assert name in self.belief_gen.keys(), f"error: agent {name} not found"

        num, text   = self.belief_gen[ name ]
        if "txt" in self.trust_mode:
            prmpt   += f"{name}: {text}\n"
        if self.trust_mode == "num":
            prmpt   += f"{name}: {num}\n"

        prmpt   += d_str[ 'post' ]
        prompt  = self.add_preamble( [ { "role": "user", "content": prmpt } ] )
        prompt  = complete.collapse_roles( prompt )

        return prompt


    def prompt_tom_entrust( self, trustee, task_desc, task_name, entrust_dialog="entrust_tom" ):
        """
        get the prompt for entrusting a trustee, using both the task specific trust belief
        where trust belief of all agents with respect of the task are stored, and the
        tom inference, stores in general trust belief

        args:
            trustee         [Trustee] the agent to entrust
            task_desc       [str] description of the task
            task_name       [str] name of the task
            entrust_dialog  [str] key in the json file of dialogs, for the appropriate choosing trustee sentence
        return:
            [str] the prompt
        """
        found       = False
        for d in self.dialog_data:
            if d[ 'id' ] == entrust_dialog:
                d_str   = d[ 'content' ]
                found   = True
                break

        if not found:
            if cnfg.VERBOSE:
                print( f"\nErrror in prompt_tom_entrust: {entrust_dialog} not found" )
            return None

        if self.trust_mode == "num":
            if cnfg.VERBOSE:
                print( f"\nErrror in prompt_tom_entrust: numeric trust_mode not available" )
            return None


        prmpt   = ''
        prmpt   += d_str[ 'intro_txt' ].format( task=task_desc )

        name        = trustee.name
        # gather both Tom and trust (if avaialble) for all trustees
        assert name in self.belief_gen.keys(), f"error: no belief found for {name}"

        _, text = self.belief_gen[ name ]
        prmpt   += f"\nFor {name} this is the inference of their attitudes:\n{text}"
        if task_name in self.belief_task[ name ].keys():
            _, text   = self.belief_task[ name ][ task_name ]
            prmpt   += f"\nand this is the turst belief in accomplishing this task:\n{text}"
        prmpt   += "\n"
        prmpt   += d_str[ 'post' ]
        prompt  = self.add_preamble( [ { "role": "user", "content": prmpt } ] )
        prompt  = complete.collapse_roles( prompt )

        return prompt
