"""
#####################################################################################################################

    HURST project - 2024


    Completions and trust evaluation

    NOTES:
    for fine-tuning: care to insert "weight": 0 in the non-final "role": "assistant", and 1 in the final one only
    insert a system role in the preliminary dialog

#####################################################################################################################
"""

import  os
import  sys
import  re
import  gc
import  platform
import  time
import  datetime
import  string
import  random
import  torch
import  numpy           as np
import  lm

frmt_response           = "%y-%m-%d_%H-%M-%S"       # datetime format for response filenames
txt_line_s              = 60 * "-" + "\n\n"         # single line separator in print text
txt_line_d              = 60 * "=" + "\n\n"         # double line separator in print text

delay                   = 120                       # delay in seconds after OpenAI internal errors

cnfg                    = None                      # configuration object
client                  = None                      # the language model client object

simulation              = None                      # the ongoing simulation object (validated by main_exec)
prompt_completions      = []                        # log of all prompts and completions
 
 
# ===================================================================================================================
#
#   Prompt manipulation
#
#       fill_placeholders
#       fill_dialog
#       chat_to_complete
#       collapse_roles
#       get_belief_dialog
#       get_dialog
#
# ===================================================================================================================

def fill_placeholders( s, d_values ):
    """
    Fill values of placeholders (if present) in the given string.
    The placeholder's name should match the key in the given dictionary.
    NOTE that if there are more keys in d_values than placeholders, it does not harm

    params:
        s:          [str] the input string
        d_values:   [dict] keys are the placeholders, values are the actual values

    return:
                    [str] filled string
    """
    # extract the placeholders' names from the formatted string
    phold   = [ v for _, v, _, _ in string.Formatter().parse( s ) if v is not None ]

    # if there are no placeholders, return the original string
    if not len( phold ):
        return s

    # check if each placeholder has a match in the given dictionary
    for v in phold:
        assert v in d_values.keys(), f"Placeholder '{v}' not found among {d_values.keys()}"

    # fill the placeholders with the dict values
    return s.format( **d_values )


def fill_dialog( dialog, d_values ):
    """
    Fill values of placeholders (if present) in all the content fields of a chat dialog
    The placeholder's name should match the key in the given dictionary.
    NOTE that if there are more keys in d_values than placeholders, it does not harm

    params:
        dialog:     [list] the input dialog
        d_values:   [dict] keys are the placeholders, values are the actual values

    return:
                    [list] filled dialog
    """

    filled  = []
    for d in dialog:
        role        = d[ "role" ]
        content     = fill_placeholders( d[ "content" ], d_values )
        filled.append( { "role": role, "content": content } )
        
    return filled


def chat_to_complete( dialog, keep_roles=True ):
    """
    convert a dialog into a single prompt

    dialog:     [list] the dialog messages
    keep_roles: [bool] include the role headers

    return:     [str] a single prompt
    """
    if isinstance( dialog, str ):       # in case of textual dialog (complete mode), there is nothing to do
        return dialog

    prompt      = ""

    for d in dialog:
        if isinstance( d, str ):
            prompt  += d
            continue
        if keep_roles:
            prompt  += d[ "role" ] +": "
        prompt  += d[ "content" ] +"\n"

    return  prompt


def collapse_roles( dialog ):
    """
    avoid consecutive duplicate roles in a dialog

    dialog:     [list] the dialog messages, or [str] for complete mode models
    one_turn:   [bool] return a dialog with just one turn

    return:     [list] the dialog messages, or [str] for complete mode models
    """

    if isinstance( dialog, str ):       # in case of textual dialog (complete mode), there is nothing to do
        return dialog

    new_dialog  = [ dialog[ 0 ] ]
    content     = ""

    role        = dialog[ 0 ][ "role" ]
    content     = dialog[ 0 ][ "content" ]

    if cnfg.one_turn:
        new_dialog[ 0 ][ "role" ] = "user"
        for d in dialog[ 1 : ]:
            sep     = '\n'
            if d[ "content" ][ 0 ] == ',':  sep = ''
            if content[ -1 ] == '\n':  sep = ''
            content = content + sep + d[ "content" ]
        new_dialog[ 0 ][ "content" ] = content
        return  new_dialog

    for i, d in enumerate( dialog[ 1 : ] ):
        if d[ "role" ] == role:
            new_dialog[ i ][ "content" ] = new_dialog[ i ][ "content" ] + '\n' + d[ "content" ]
        else:
            role        = d[ "role" ]
            content     = d[ "content" ]
            new_dialog.append( d )

    return  new_dialog


def get_belief_dialog( title, data, trust_mode="txt", var=None ):
    """
    get dialogue for trust belief updates

    title:      [str] string with the title of the dialog
    data:       [dict] result of json5.load of file f_dialog
    trust_mode: [str] the way trust is managed
    var:        [dict] with variable for placeholders

    return:     [list] the dialog messages or [str] for complete mode models
    """
    if "txt" in trust_mode:
        trust_mode  = "txt"
    d_titles    = [ d[ 'id' ] for d in data ]

    if title not in d_titles:
        if cnfg.VERBOSE:
            print( f"Error in get_belief_dialog: no title {title} found in json file {fname}" )
        return None
    idx     = d_titles.index( title )
    content = data[ idx ][ "content" ]
    if trust_mode not in content.keys():
        if cnfg.VERBOSE:
            print( f"Error in get_belief_dialog: trust mode {trust_mode} not found in content {content}" )
        return None
    text        = content[ trust_mode ][ "content" ]
    if var is not None:
        text    = fill_placeholders( text, var )

    prompt      = [ { "role": "user", "content": text } ]
    return  prompt


def get_dialog( titles, data, intro=None ):
    """
    get preliminary dialogues for chat and complete models

    titles:     [list] of [str] strings with the title of the dialogues, [] for all titles in data
    data:       [dict] result of json5.load of file f_dialog, or a file with similar structure
    intro       [list] dialog with the agent's self-introduction, or None, ignored when titles=[]

    return:     [list] the dialog messages or string for completion mode
    """
    if not len( titles ):
        prompt  = []
        for d in data:
            prompt      += d[ "content" ]
        return  prompt

    d_titles    = [ d[ 'id' ] for d in data ]
    title       = titles[ 0 ]
    try:
        idx     = d_titles.index( title )
    except Exception as e:
        print( f"non existing dialog with title {title}" )
        raise e
    prompt      = data[ idx ][ "content" ]
    if intro is not None:
        if "mode" in data[ idx ].keys():
            if data[ idx ][ "mode" ] == "after_name":
                prompt  = intro + prompt
            else:
                prompt  = prompt + intro
        else:
            prompt  = prompt + intro

    if len( titles ) > 1:
        for title in titles[ 1 : ]:
            try:
                idx     = d_titles.index( title )
            except Exception as e:
                print( f"non existing dialog with title {title}" )
                raise e
            prompt      += data[ idx ][ "content" ]

    return  prompt



# ===================================================================================================================
#
#   Manage completions
#
#       complete_openai
#       complete_anthro
#       complete_hf
#       complete
#       extract_trust
#       extract_decision
#
# ===================================================================================================================

def complete_openai( prompt ):
    """
    Feed a prompt to a model, and return the list of completions.
    It works either with models with straight completion and models with chat completion.
    Manage the first possible OpenAI failure, simply by waiting

    params:
        prompt:     the prompt [str] or the messages [list] for models with chat completion

    return:
                    [list] with completions [str]
    """
    global client
    assert isinstance( prompt, list ), "For chat complete models, the prompt should be a list"

    # check if openai has already a client, otherwise set it
    if client is None:
        client  = lm.set_openai()
    user    = os.getlogin() + '@' + platform.node()

    # arguments for completion calls
    cargs   = {
            "model"             : cnfg.model,
            "max_tokens"        : cnfg.max_tokens,
            "n"                 : cnfg.n_returns,
            "top_p"             : cnfg.top_p,
            "temperature"       : cnfg.temperature,
            "stop"              : None,
            "user"              : user,
            "messages"          : prompt
    }

    try:                                # recently OpenAI fails sometimes, for service_unavailable_error...
        res     = client.chat.completions.create( **cargs )
    except Exception as e:              # catch EVERY exception to ensure compatibility with OpenAI versions
        if cnfg.VERBOSE:
            print( f"catched error {e}, sleeping {delay} seconds and trying again" )
        time.sleep( delay )
        res     = client.chat.completions.create( **cargs )
    return [ t.message.content for t in res.choices ]


def complete_anthro( prompt ):
    """
    Feed a prompt to an anthropic model and get the list of completions returned.

    params:
        prompt      [str] or [list] the prompt for completion-mode models,
                    or the messages for chat-mode models

    return:         [list] with completions [str]
    """
    global client
    if cnfg.DEBUG:  return [ "test_only" ]

    if client is None:              # check if anthropic has already a client, otherwise set it
        client  = lm.set_anthro()
 
    # arguments for completion calls
    cargs   = {
            "messages"          : prompt,
            "model"             : cnfg.model,
            "max_tokens"        : cnfg.max_tokens,
            "top_p"             : cnfg.top_p,
            "temperature"       : cnfg.temperature,
    }

    # there have been anthropic._exceptions.OverloadedError errors, this is a very simple workaround
    # if the problem will repeat more often, and is not solved, the solution is exponential backoff
    try:
        res     = client.messages.create( **cargs )
    except Exception as e:              # catch EVERY exception to ensure compatibility with OpenAI/anthropic versions
        if cnfg.VERBOSE:
            print( f"catched error {e}, sleeping {delay} seconds and trying again" )
        time.sleep( delay )             # just sleep for a while and then try again
        res     = client.messages.create( **cargs )
    return [ res.content[ 0 ].text ]


def complete_hf( prompt ):
    """
    Feed a prompt to a model, and return the list of completions.
    It works either with models with straight completion and models with chat completion.

    params:
        prompt:     the prompt [str] or the messages [list] for models with chat completion

    return:
                    [list] with completions [str]
    """
    global client

    # check if hugginface has already a client, otherwise set it
    if client is None:
        client  = lm.set_hf()

    cargs   = {
            "max_new_tokens"        : cnfg.max_tokens,
            "return_full_text"      : False,
            "num_return_sequences"  : cnfg.n_returns,
            "top_p"                 : cnfg.top_p,
            "temperature"           : cnfg.temperature,
            "do_sample"             : True,
    }
    res         = client( prompt, **cargs )
    completions = [ t[ "generated_text" ] for t in res ]
    if "fine_tuned" in cnfg.model:
        completions = [ clean_completion( c, drastic=True ) for c in completions ]

    return completions


def complete( prompt ):
    """
    Feed a prompt to a model, and return the list of completions.
    It works either with models with straight completion and models with chat completion.

    params:
        prompt:     the prompt [str] or the messages [list] for models with chat completion

    return:
                    [list] with completions [str]
    """
    global client

    if cnfg.DEBUG:
        completion  = [ "test_only" ]
        if simulation is not None:
            clock           = simulation.get_clock()
            boostrap_stage  = simulation.boostrap_stage
        else:
            clock           = 0
            boostrap_stage  = False
        prompt_completions.append( [ clock, boostrap_stage, prompt, completion, 0 ] )
        return completion

    t_start     = datetime.datetime.now()
    match cnfg.interface:
        case 'openai':
            completion  = complete_openai( prompt )
        case 'anthro':
            completion  = complete_anthro( prompt )
        case 'llama':
            completion  = complete_llama( prompt )
        case 'hf':
            completion  = complete_hf( prompt )
        case _:
            print( f"model interface '{cnfg.interface}' not supported" )
            return None
    t_end       = datetime.datetime.now()

    if simulation is not None:
        clock           = simulation.get_clock()
        boostrap_stage  = simulation.boostrap_stage
    else:
        clock           = 0
        boostrap_stage  = False
    prompt_completions.append( [ clock, boostrap_stage, prompt, completion, t_end - t_start ] )

    return completion


def extract_trust( compl ):
    """
    Extract the numerical score from a sentence similar to "I would rate Agent a 3 out of 5."

    params:
        compl   [str] sentence containing the numerical score

    return:
                [int] trust score, None if score is absent or ambiguous
    """
    numbers = []

    # remove dots, commas and other signs otherwise words like "3." are not recognized as numbers
    for char in ".,/-=":
        compl   = compl.replace( char, ' ' )
    words   = compl.split()

    # get numerical values in the sentence
    for w in words:
        if w.isnumeric():
            numbers.append( int( w ) )

    match len( numbers ):
        case 0:
            if cnfg.VERBOSE: print( f"There are no scores in the sentence '{compl}'" )
            return None

        case 1 | 2:
            return min( numbers )

        case 3:
            return int( np.median( numbers ) )

        case _:
            if cnfg.VERBOSE: print( f"Ambiguous score in the sentence '{compl}'" )
            return None


def extract_decision( compl ):
    """
    extract from the completion the decision to entrust or not an agent
    the function is simply, without any interpretation of affermative/negative sentences,
    therefore it relies on a strict compliance of the model with the completion format
    an exception handled is the two component reasoning/answer of models like gpt-oss-20b,
    where a specifi marker separates the two components

    params:
        compl   [str] sentence containing the decision taken

    return:
                [bool] the decision
    """
    compl   = compl.lower()

    # this is the special case of models like gpt-oss-20b that use a special marker for the final response
    marker  = "assistantfinal"
    if marker in compl:
        an, fin = compl.split( "assistantfinal", 1 )
        compl   = fin

    if "yes" in compl:
        return True
    return False


def extract_response( compl ):
    """
    extract from the completion a response, either boolean of in Likert scale
    the function is simply, without elaborate cleaning therefore it relies on a strict compliance
    of the model with the completion format an exception handled is the two component reasoning/answer
    of models like gpt-oss-20b, where a specifi marker separates the two components
    NOTE: there is a problem of returning 0 for Likert scale completions, should be fixed

    params:
        compl   [str] sentence containing the decision taken

    return:
                [int] the response
    """
    compl   = compl.lower()

    # this is the special case of models like gpt-oss-20b that use a special marker for the final response
    marker  = "assistantfinal"
    if marker in compl:
        an, fin = compl.split( "assistantfinal", 1 )
        compl   = fin

    if "yes" in compl:
        return 1
    if "no" in compl:
        return 0
    for n in range( 1, 8 ):
        if str( n ) in compl:
            return n

    return 0


def clean_assessment( completion ):
    """
    clean the completion of a trust/ToM assessment, pruning unnecessary information, and taking into
    account the use a special marker for the final response, as in gpt-oss-20b

    params:
        completion  [str] sentence containing the revused assessment

    return:
                    [str] the cleaned completion
    """
    # this is the special case of models like gpt-oss-20b that use a special marker for analisys, and
    # response
    start_mark  = "analysis"
    end_mark    = "assistantfinal"
    # there can be different patterns, as observed in completions from Llama3.1, Llama2-7, GPT4omini
    # this list can be expanded after more extensive observation of the logs
    patterns    = [
        "[Bb]ased on.* reassess.* as follows:",
        "[Bb]ased on.* reassess my inference positively."
    ]
    skip    = ":;.\n"                             # possible useless characters after the pattern to prune

    if end_mark in completion:
        an, fin     = completion.split( "assistantfinal", 1 )
        completion  = fin

    found   = False
    for pattern in patterns:
        found   = re.search( pattern, completion )
        if found:
            _, i        = found.span()
            while completion[ i ] in skip:
                i       += 1
            completion  = completion[ i : ]

    if completion.startswith( start_mark ):     # case where in gpt-oss-20b like models the analyis is too long
        return None

    return completion


# ===================================================================================================================
#
#   Write results
#
# ===================================================================================================================

def print_header( fstream ):
    """
    Print on file information about the model and execution.
    The name of the file is automatically generated with current date and time.

    args:
         fstream   [TextIOWrapper] text stream
    """
    command     = sys.executable + " " + " ".join( sys.argv )
    host        = platform.node()

    fstream.write( txt_line_d )
    fstream.write( "executing:\n" + command )     # write the command line that executed the completion
    fstream.write( "\non host " + host + "\n\n" )
    fstream.write( txt_line_d )
    fstream.write( str( cnfg ) + "\n" )           # write all information on parameters used in the completion
    fstream.write( txt_line_d )


def print_content( fstream, clock, boostrap, prompt, completion, duration ):
    """
    Print on file the content of a prompt and its completions

    params:
        fstream     [TextIOWrapper] text stream of the output file
        boostrap    [bool] simulation in boostrap stage
        clock       [int] of the simulation
        prompt      [list] of dialog messages or [str]
        completion  [list] of [str]
    """
    p_len   = 0
    fstream.write( txt_line_s )
    if boostrap:
        fstream.write( f"at simulation time {clock} - boostrap stage\n" )
    else:
        fstream.write( f"at simulation time {clock} - full capacity stage\n" )

    if isinstance( prompt, str ):
        fstream.write( f"\nPROMPT:\n{prompt}\n" )
    else:
        for p in prompt:
            fstream.write( f"\n{p['role'].upper()}:\n" )
            content         = p[ "content" ]
            p_len           += len( content )
            fstream.write( f"{content}\n" )

    fstream.write( "\n" + txt_line_s )

    for i, c in enumerate( completion ):
        fstream.write( f"Completion #{i:02d}:\n{c}\n\n" )

    fstream.write( f"\nPrompt length:   {p_len} words" )
    fstream.write( f"\nCompletion time: {duration}\n\n" )
    fstream.write( txt_line_d )


def print_simulation( fstream ):
    """
    Print on file the content of all prompts and completions

    params:
        fstream     [TextIOWrapper] text stream of the output file
    """
    fstream.write( '\n\n' + txt_line_d )
    fstream.write( "complete report of prompts and completions during the simulation\n" )
    fstream.write( txt_line_d )
    for pc in prompt_completions:
        print_content( fstream, *pc )

    fstream.write( txt_line_d )
