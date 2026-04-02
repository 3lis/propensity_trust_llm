"""
#####################################################################################################################

    HURST project - 2024

    Handling logics for ideal trust update and partner choice

#####################################################################################################################
"""

import  re
from    complete        import fill_placeholders

rule_codes          = ( 0, 0 )

N_LEVEL             = None
levels_prop         = None

# "unknown" must always be at index 0
levels_prop_5       = [ "unknown", "very low", "low", "medium", "high", "very high" ]
levels_prop_3       = [ "unknown", "low", "medium", "high" ]

prompt_belief       = "Capability: {comp}. Reliability: {reli}. Willingness: {will}."
prompt_choice       = "I choose {name} because I trust them to have the required level of the properties needed for the task, which are: {prop1} ({val1}) and {prop2} ({val2})."

task_prop_names     = [ "competence", "reliability", "willingness" ]
clean_prop_names    = [ "Capability", "Reliability", "Willingness" ]


def get_task_props( task ):
    """
    Return ordered properties of a task

    param:
        task    [dict]

    return:
        [list] of [str] in decreasing order
    """
    task_prop   = task[ 'prop' ]
    p1          = [ k for k, v in task_prop.items() if v == 4 ][ 0 ].capitalize()
    p2          = [ k for k, v in task_prop.items() if v == 2 ][ 0 ].capitalize()
    p3          = [ k for k, v in task_prop.items() if v == 1 ][ 0 ].capitalize()
    p1          = "Capability" if p1 == "Competence" else p1
    p2          = "Capability" if p2 == "Competence" else p2
    p3          = "Capability" if p3 == "Competence" else p3
    return p1, p2, p3


def belief_to_str( d_belief ):
    """
    Convert a dict of belief to string for prompt
    """
    p       = { "comp": levels_prop[ d_belief[ clean_prop_names[ 0 ] ] ],
                "reli": levels_prop[ d_belief[ clean_prop_names[ 1 ] ] ],
                "will": levels_prop[ d_belief[ clean_prop_names[ 2 ] ] ] }
    return fill_placeholders( prompt_belief, p )


def belief_to_dict( s_belief ):
    """
    Convert a string of belief to dict with numerical values
    """
    if s_belief is None:
        d   = { clean_prop_names[ 0 ]: levels_prop.index( "unknown" ),
                clean_prop_names[ 1 ]: levels_prop.index( "unknown" ),
                clean_prop_names[ 2 ]: levels_prop.index( "unknown" ) }
    else:
        s   = s_belief.replace( '.', '' ).replace( ':', '' ).split()
        s   = re.split( r'[:.]', s_belief )
        s   = [ x.strip() for x in s if x.strip() ]
        d   = { s[ 0 ]: levels_prop.index( s[ 1 ] ),
                s[ 2 ]: levels_prop.index( s[ 3 ] ),
                s[ 4 ]: levels_prop.index( s[ 5 ] ) }
    return d


#####################################################################################################################


def rule_TU_1( task, succ, fail, belief_1s, idim=None ):
    """
    Logic of trust update:
        - a task requires especially X but also enough Y
        - if there is a success     --> X++ and Y++ (or X=high and Y=medium if unknown)
        - if there is a fail on X   --> X--         (or X=low if unknown)
        - if there is a fail on Y   --> X++ and Y-- (or X=high and Y=low if unknown)

    param:
        task        [dict]
        succ        [bool]
        fail        [int] index of prop in case of succ=False
        belief_1s   [str] previous belief as string

    return:
        [str] assistant prompt output
    """
    p1, p2, p3      = get_task_props( task )
    belief_1d       = belief_to_dict( belief_1s )
    belief_2d       = {}
    belief_2d[ p3 ] = belief_1d[ p3 ]    # the last prop remains the same

    if succ:
        # success
        if belief_1d[ p1 ]  == levels_prop.index( "unknown" ):
            # case "unknown" for main prop
            belief_2d[ p1 ]     = levels_prop.index( "high" )
        else:
            # increase main prop
            belief_2d[ p1 ]     = min( len( levels_prop )-1, belief_1d[ p1 ]+1 )

        if belief_1d[ p2 ]  == levels_prop.index( "unknown" ):
            # case "unknown" for second prop
            belief_2d[ p2 ]     = levels_prop.index( "medium" )
        else:
            # increase second prop
            belief_2d[ p2 ]     = min( len( levels_prop )-1, belief_1d[ p2 ]+1 )

    elif clean_prop_names[ fail ] == p1:
        # fail on main prop
        belief_2d[ p2 ] = belief_1d[ p2 ]    # the second prop remains the same
        if belief_1d[ p1 ]  == levels_prop.index( "unknown" ):
            # case "unknown"
            belief_2d[ p1 ]     = levels_prop.index( "low" )
        else:
            # decrease
            belief_2d[ p1 ]     = max( 1, belief_1d[ p1 ]-1 )

    elif clean_prop_names[ fail ] == p2:
        # fail on secondary prop
        if belief_1d[ p1 ] == levels_prop.index( "unknown" ):
            # case "unknown" for main prop
            belief_2d[ p1 ]  = levels_prop.index( "high" )
        else:
            # increase main prop
            belief_2d[ p1 ]  = min( len( levels_prop )-1, belief_1d[ p1 ]+1 )

        if belief_1d[ p2 ] == levels_prop.index( "unknown" ):
            # case "unknown" for second prop
            belief_2d[ p2 ]  = levels_prop.index( "low" )
        else:
            # decrease second prop
            belief_2d[ p2 ]  = max( 1, belief_1d[ p2 ]-1 )

    elif clean_prop_names[ fail ] == p3:
        print( "WARNING: this case should never happen! Something is wrong..." )
        return

    # special case of examples for one property only, leaving all the other unknown
    if idim is not None:
        dim     = clean_prop_names[ idim ]
        for d in belief_2d.keys():
            if d != dim:
                belief_2d[ d ]  = levels_prop.index( "unknown" )

    belief_2s   = belief_to_str( belief_2d )
    return belief_2s


def rule_TU_1_DPO( task, succ, fail, belief_1s, idim=None ):
    """
    Exact copy of rule_TU_1 (not very elegant) with the addition on "right" (chosen)
    and "wrong" (rejected) samples.

    Logic of right trust update:
        - a task requires especially X (main prop) but also enough Y (second prop)
        - if there is a success     --> X++ and Y++ (or X=high and Y=medium if unknown)
        - if there is a fail on X   --> X--         (or X=low if unknown)
        - if there is a fail on Y   --> X++ and Y-- (or X=high and Y=low if unknown)

    Logic of wrong trust update:
        - a task requires especially X (main prop) but also enough Y (second prop)
        - if there is a success     --> X-- and Y-- (or X=low and Y=low if unknown)
        - if there is a fail on X   --> X++         (or X=high if unknown)
        - if there is a fail on Y   --> X-- and Y++ (or X=low and Y=high if unknown)

    param:
        task        [dict]
        succ        [bool]
        fail        [int] index of prop in case of succ=False
        belief_1s   [str] previous belief as string

    return:
        [list] of [str] "chosen" prompt and "rejected" prompt
    """
    p1, p2, p3      = get_task_props( task )
    belief_1d       = belief_to_dict( belief_1s )   # previous belief as dict
    belief_2d       = {}                            # new (right) chosen belief
    belief_3d       = {}                            # new (wrong) rejected belief

    belief_2d[ p3 ] = belief_1d[ p3 ]               # the last prop remains the same
    belief_3d[ p3 ] = belief_1d[ p3 ]               # the last prop remains the same

    if succ:
        # success
        if belief_1d[ p1 ]  == levels_prop.index( "unknown" ):
            # case "unknown" for main prop
            belief_2d[ p1 ]     = levels_prop.index( "high" )
            belief_3d[ p1 ]     = levels_prop.index( "low" )
        else:
            # increase main prop for right case
            belief_2d[ p1 ]     = min( len( levels_prop )-1, belief_1d[ p1 ]+1 )
            # decrease main prop for wrong case
            belief_3d[ p1 ]     = max( 1, belief_1d[ p1 ]-1 )

        if belief_1d[ p2 ]  == levels_prop.index( "unknown" ):
            # case "unknown" for second prop
            belief_2d[ p2 ]     = levels_prop.index( "medium" )
            belief_3d[ p2 ]     = levels_prop.index( "low" )
        else:
            # increase second prop for right case
            belief_2d[ p2 ]     = min( len( levels_prop )-1, belief_1d[ p2 ]+1 )
            # decrease second prop for wrong case
            belief_3d[ p2 ]     = max( 1, belief_1d[ p2 ]-1 )

    elif clean_prop_names[ fail ] == p1:
        # fail on main prop
        belief_2d[ p2 ] = belief_1d[ p2 ]    # the second prop remains the same
        belief_3d[ p2 ] = belief_1d[ p2 ]    # the second prop remains the same
        if belief_1d[ p1 ]  == levels_prop.index( "unknown" ):
            # case "unknown"
            belief_2d[ p1 ]     = levels_prop.index( "low" )
            belief_3d[ p1 ]     = levels_prop.index( "high" )
        else:
            # right decrease
            belief_2d[ p1 ]     = max( 1, belief_1d[ p1 ]-1 )
            # wrong increase
            belief_3d[ p1 ]     = min( len( levels_prop )-1, belief_1d[ p1 ]+1 )

    elif clean_prop_names[ fail ] == p2:
        # fail on secondary prop
        if belief_1d[ p1 ] == levels_prop.index( "unknown" ):
            # case "unknown" for main prop
            belief_2d[ p1 ]  = levels_prop.index( "high" )
            belief_3d[ p1 ]  = levels_prop.index( "low" )
        else:
            # right increase main prop
            belief_2d[ p1 ]  = min( len( levels_prop )-1, belief_1d[ p1 ]+1 )
            # wrong decrease main prop
            belief_3d[ p1 ]  = max( 1, belief_1d[ p1 ]-1 )

        if belief_1d[ p2 ] == levels_prop.index( "unknown" ):
            # case "unknown" for second prop
            belief_2d[ p2 ]  = levels_prop.index( "low" )
            belief_3d[ p2 ]  = levels_prop.index( "high" )
        else:
            # right decrease second prop
            belief_2d[ p2 ]  = max( 1, belief_1d[ p2 ]-1 )
            # wrong increase second prop
            belief_3d[ p2 ]  = min( len( levels_prop )-1, belief_1d[ p2 ]+1 )

    elif clean_prop_names[ fail ] == p3:
        print( "WARNING: this case should never happen! Something is wrong..." )
        return

    # special case of examples for one property only, leaving all the other unknown
    if idim is not None:
        dim     = clean_prop_names[ idim ]
        for d in belief_2d.keys():
            if d != dim:
                belief_2d[ d ]  = levels_prop.index( "unknown" )
                belief_3d[ d ]  = levels_prop.index( "unknown" )

    belief_chosen   = belief_to_str( belief_2d )
    belief_rejected = belief_to_str( belief_3d )
    return belief_chosen, belief_rejected


def rule_TU_2( task, succ, fail, belief_1s, idim=None ):
    """
    Logic of trust update:
        - a task requires especially X but also enough Y
        - if there is a success     --> X=high and Y=medium
        - if there is a fail on X   --> X=medium
        - if there is a fail on Y   --> X=high and Y=low

    param:
        task        [dict]
        succ        [bool]
        fail        [int] index of prop in case of succ=False
        belief_1s   [str]

    return:
        [str] assistant prompt output
    """
    p1, p2, p3      = get_task_props( task )
    belief_1d       = belief_to_dict( belief_1s )
    belief_2d       = {}
    belief_2d[ p3 ] = belief_1d[ p3 ]    # the last prop remains the same

    if succ:
        # success
        belief_2d[ p1 ]     = levels_prop.index( "high" )
        belief_2d[ p2 ]     = levels_prop.index( "medium" )

    elif clean_prop_names[ fail ] == p1:
        # fail on main prop
        belief_2d[ p1 ]     = levels_prop.index( "medium" )
        belief_2d[ p2 ]     = belief_1d[ p2 ]    # the second prop remains the same

    elif clean_prop_names[ fail ] == p2:
        # fail on secondary prop
        belief_2d[ p1 ]     = levels_prop.index( "high" )
        belief_2d[ p2 ]     = levels_prop.index( "low" )

    elif clean_prop_names[ fail ] == p3:
        print( "WARNING: this case should never happen! Something is wrong..." )
        return

    # special case of examples for one property only, leaving all the other unknown
    if idim is not None:
        dim     = clean_prop_names[ idim ]
        for d in belief_2d.keys():
            if d != dim:
                belief_2d[ d ]  = levels_prop.index( "unknown" )

    belief_2s   = belief_to_str( belief_2d )
    return belief_2s


def rule_TU_3( task, succ, fail, belief_1s, idim=None ):
    """
    Logic of trust update:
        - a task requires especially X but also enough Y
        - if there is a success     --> X must be high or above
                                        Y must be medium or above
        - if there is a fail on X   --> X must be medium or below
        - if there is a fail on Y   --> X must be high or above
                                        Y must be low or below
    param:
        task        [dict]
        succ        [bool]
        fail        [int] index of prop in case of succ=False
        belief_1s   [str]

    return:
        [str] assistant prompt output
    """
    p1, p2, p3      = get_task_props( task )
    belief_1d       = belief_to_dict( belief_1s )
    belief_2d       = {}
    belief_2d[ p3 ] = belief_1d[ p3 ]    # the last prop remains the same

    if succ:
        # success
        if belief_1d[ p1 ] < levels_prop.index( "high" ) or belief_1d[ p1 ] == 0:
            # main prop must be high or above
            belief_2d[ p1 ]     = levels_prop.index( "high" )
        else:
            belief_2d[ p1 ]     = belief_1d[ p1 ]

        if belief_1d[ p2 ] < levels_prop.index( "medium" ) or belief_1d[ p2 ] == 0:
            # second prop must be medium or above
            belief_2d[ p2 ]     = levels_prop.index( "medium" )
        else:
            belief_2d[ p2 ]     = belief_1d[ p2 ]

    elif clean_prop_names[ fail ] == p1:
        # fail on main prop
        belief_2d[ p2 ] = belief_1d[ p2 ]    # the second prop remains the same

        if belief_1d[ p1 ] > levels_prop.index( "medium" ) or belief_1d[ p1 ] == 0:
            # main prop must be medium or below
            belief_2d[ p1 ]     = levels_prop.index( "medium" )
        else:
            belief_2d[ p1 ]     = belief_1d[ p1 ]

    elif clean_prop_names[ fail ] == p2:
        # fail on secondary prop
        if belief_1d[ p1 ] < levels_prop.index( "high" ) or belief_1d[ p1 ] == 0:
            # main prop must be high or above
            belief_2d[ p1 ]     = levels_prop.index( "high" )
        else:
            belief_2d[ p1 ]     = belief_1d[ p1 ]

        if belief_1d[ p2 ] > levels_prop.index( "low" ) or belief_1d[ p2 ] == 0:
            # second prop must be low or below
            belief_2d[ p2 ]     = levels_prop.index( "low" )
        else:
            belief_2d[ p2 ]     = belief_1d[ p2 ]

    elif clean_prop_names[ fail ] == p3:
        print( "WARNING: this case should never happen! Something is wrong..." )
        return

    # special case of examples for one property only, leaving all the other unknown
    if idim is not None:
        dim     = clean_prop_names[ idim ]
        for d in belief_2d.keys():
            if d != dim:
                belief_2d[ d ]  = levels_prop.index( "unknown" )

    belief_2s   = belief_to_str( belief_2d )
    return belief_2s


def rule_TU_4( task, succ, fail, belief_1s, idim=None ):
    """
    Logic of trust update:
        - a task requires especially X
        - if there is a success     --> X++ (or X=high if unknown)
        - if there is a fail on X   --> X-- (or X=low if unknown)
        - if there is a fail on Y   --> nothing happens

    param:
        task        [dict]
        succ        [bool]
        fail        [int] index of prop in case of succ=False
        belief_1s   [str]

    return:
        [str] assistant prompt output
    """
    p1, p2, p3      = get_task_props( task )
    belief_1d       = belief_to_dict( belief_1s )
    belief_2d       = {}

    belief_2d[ p2 ] = belief_1d[ p2 ]    # the last two props remain the same
    belief_2d[ p3 ] = belief_1d[ p3 ]

    if succ:
        # success
        if belief_1d[ p1 ]  == levels_prop.index( "unknown" ):
            # case "unknown" for main prop
            belief_2d[ p1 ]     = levels_prop.index( "high" )
        else:
            # increase main prop
            belief_2d[ p1 ]     = min( len( levels_prop )-1, belief_1d[ p1 ]+1 )

    elif clean_prop_names[ fail ] == p1:
        # fail on main prop
        if belief_1d[ p1 ]  == levels_prop.index( "unknown" ):
            # case "unknown"
            belief_2d[ p1 ]     = levels_prop.index( "low" )
        else:
            # decrease
            belief_2d[ p1 ]     = max( 1, belief_1d[ p1 ]-1 )

    elif clean_prop_names[ fail ] == p2:
        # fail on secondary prop
        belief_2d[ p1 ] = belief_1d[ p1 ]   # the main prop remains the same

    elif clean_prop_names[ fail ] == p3:
        print( "WARNING: this case should never happen! Something is wrong..." )
        return

    # special case of examples for one property only, leaving all the other unknown
    if idim is not None:
        dim     = clean_prop_names[ idim ]
        for d in belief_2d.keys():
            if d != dim:
                belief_2d[ d ]  = levels_prop.index( "unknown" )

    belief_2s   = belief_to_str( belief_2d )
    return belief_2s


def rule_PC_1( task, dict_beliefs_s ):
    """
    Logic of partner choice:
        - a task requires especially X but also enough Y
        - pick the agent with highest X
        - if there is a tie, pick the agent with highest Y

    param:
        task            [dict]
        dict_beliefs_s  [dict] keys are agent names, values are [str]

    return:
        [str] assistant prompt output
    """
    p1, p2, p3  = get_task_props( task )
    partners    = list( dict_beliefs_s.keys() )
    beliefs     = list( dict_beliefs_s.values() )
    beliefs     = [ belief_to_dict( s ) for s in beliefs ]

    # select the element with highest prop1, and if there's a tie check prop2
    b_best  = max( beliefs, key=lambda b: ( b[ p1 ], b[ p2 ] ) )
    i_best  = beliefs.index( b_best )

    p       = { "name":     partners[ i_best ],
                "prop1":    p1,
                "val1":     levels_prop[ b_best[ p1 ] ],
                "prop2":    p2,
                "val2":     levels_prop[ b_best[ p2 ] ] }
    res     = fill_placeholders( prompt_choice, p )
    return res


def rule_PC_1_DPO( task, dict_beliefs_s ):
    """
    Exact copy of rule_PC_1 (not very elegant) with the addition on "right" (chosen)
    and "wrong" (rejected) samples.

    Logic of right partner choice:
        - a task requires especially X (main prop) but also enough Y (second prop)
        - pick the agent with highest X
        - if there is a tie, pick the agent with highest Y

    Logic of wrong partner choice:
        - a task requires especially X (main prop) but also enough Y (second prop)
        - pick the agent with lowest X
        - if there is a tie, pick the agent with lowest Y

    param:
        task            [dict]
        dict_beliefs_s  [dict] keys are agent names, values are [str]

    return:
        [list] of [str] "chosen" prompt and "rejected" prompt
    """
    p1, p2, p3  = get_task_props( task )
    partners    = list( dict_beliefs_s.keys() )
    beliefs     = list( dict_beliefs_s.values() )
    beliefs     = [ belief_to_dict( s ) for s in beliefs ]

    # select the element with highest prop1, and if there's a tie check prop2
    b_best_right    = max( beliefs, key=lambda b: ( b[ p1 ], b[ p2 ] ) )
    i_best_right    = beliefs.index( b_best_right )

    # select the element with lowest prop1, and if there's a tie check prop2
    b_best_wrong    = min( beliefs, key=lambda b: ( b[ p1 ], b[ p2 ] ) )
    i_best_wrong    = beliefs.index( b_best_wrong )

    p       = { "name":     partners[ i_best_right ],
                "prop1":    p1,
                "val1":     levels_prop[ b_best_right[ p1 ] ],
                "prop2":    p2,
                "val2":     levels_prop[ b_best_right[ p2 ] ] }

    q       = { "name":     partners[ i_best_wrong ],
                "prop1":    p1,
                "val1":     levels_prop[ b_best_wrong[ p1 ] ],
                "prop2":    p2,
                "val2":     levels_prop[ b_best_wrong[ p2 ] ] }

    belief_chosen   = fill_placeholders( prompt_choice, p )
    belief_rejected = fill_placeholders( prompt_choice, q )

    return belief_chosen, belief_rejected


def rule_PC_2( task, dict_beliefs_s ):
    """
    Logic of partner choice:
        - a task requires especially X but also enough Y
        - pick the agent with highest weighted mean of X and Y

    param:
        task            [dict]
        dict_beliefs_s  [dict] keys are agent names, values are [str]

    return:
        [str] assistant prompt output
    """
    w1, w2      = ( 0.6, 0.4 )
    p1, p2, p3  = get_task_props( task )
    partners    = list( dict_beliefs_s.keys() )
    beliefs     = list( dict_beliefs_s.values() )
    beliefs     = [ belief_to_dict( s ) for s in beliefs ]

    b_best      = -1
    i_best      = -1
    for i, b in enumerate( beliefs ):
        m       = w1 * b[ p1 ] + w2 * b[ p2 ]
        if m > b_best:
            b_best  = m
            i_best  = i

    p       = { "name":     partners[ i_best ],
                "prop1":    p1,
                "val1":     levels_prop[ beliefs[ i_best ][ p1 ] ],
                "prop2":    p2,
                "val2":     levels_prop[ beliefs[ i_best ][ p2 ] ] }
    res     = fill_placeholders( prompt_choice, p )
    return res


#####################################################################################################################


def set_levels_prop():
    """
    Set number of prop levels according to the decimal values of rule_codes
    """
    global N_LEVEL, levels_prop

    code_TU, code_PC    = rule_codes
    decim_TU            = int( str( code_TU )[ -1 ] )
    decim_PC            = int( str( code_PC )[ -1 ] )
    assert decim_TU == decim_PC, f"code_TU and code_PC should indicate the same number of prop levels"

    match decim_TU:
        case 0 | 3:
            N_LEVEL     = 3
            levels_prop = levels_prop_3
        case 5:
            N_LEVEL     = 5
            levels_prop = levels_prop_5
        case _:
            print( f"ERROR: case \"{code_TU}\" is not handled yet in set_levels_prop()..." )

    return N_LEVEL, levels_prop


def rule_TU( task, succ, fail, belief_1s, idim=None ):
    """
    General function for trust update

    param:
        task        [dict]
        succ        [bool]
        fail        [int] index of prop in case of succ=False
        belief_1s   [str]

    return:
        [str] assistant prompt output
    """
    code_TU     = rule_codes[ 0 ]
    res         = None
    set_levels_prop()

    match code_TU:
        case 0 | 1.3 | 1.5:
            res         = rule_TU_1( task, succ, fail, belief_1s, idim=idim )
        case 2.3 | 2.5:
            res         = rule_TU_2( task, succ, fail, belief_1s, idim=idim )
        case 3.3 | 3.5:
            res         = rule_TU_3( task, succ, fail, belief_1s, idim=idim )
        case 4.3 | 4.5:
            res         = rule_TU_4( task, succ, fail, belief_1s, idim=idim )
        case _:
            print( f"ERROR: case \"{code_TU}\" is not handled yet in rule_TU()..." )

    return res


def rule_PC( task, dict_beliefs_s ):
    """
    General function for partner choice

    param:
        task            [dict]
        dict_beliefs_s  [dict] keys are agent names, values are [str]

    return:
        [str] assistant prompt output
    """
    code_PC     = rule_codes[ 1 ]
    res         = None
    set_levels_prop()

    match code_PC:
        case 0 | 1.3 | 1.5:
            res         = rule_PC_1( task, dict_beliefs_s )
        case 2.3 | 2.5:
            res         = rule_PC_2( task, dict_beliefs_s )
        case _:
            print( f"ERROR: case \"{rule_PC}\" is not handled yet in rule_PC()..." )

    return res
