"""
#####################################################################################################################

    trust propensity project - 2025

    utility for collecting results of Frazier questions
    just print the results, formatted in LaTeX

#####################################################################################################################
"""

import  os
import  sys

res         = "../res"
log         = "runs.log"

first       = "25-08-31_11-16-50"
last        = "25-08-31_12-04-47"
first       = "25-08-31_18-03-40"
last        = "25-08-31_18-56-53"
n_items     = 12

def get_info ( lines ):
    """
    retrieve essential info
    """

    conf        = lines[ : 55 ]     # the configuration part
    found       = False
    i           = 10
    while not found:
        l       = conf[ i ]
        if l.startswith( "model_short" ):
            string      = l.split()[ -1 ]
            found       = True
        i       += 1
        if i > 60:
            return None

    found       = False
    while not found:
        l       = conf[ i ]
        if "Frazier" in l:
            found       = True
        i       += 1
        if i > 60:
            return None

    i0      = i + 1
    for i in range( i0, i0 + n_items ):
        l       = conf[ i ]
        string  += " & "
        string  += l.split()[ -1 ]

    found       = False
    while not found:
        l       = conf[ i ]
        if l.startswith( "overall" ):
            string  += " & "
            string  += l.split()[ -1 ]
            found       = True
        i       += 1
        if i > 60:
            return None

    string      += r"\\"

    return string


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    list_res    = sorted( os.listdir( res ) )

    i_first     = list_res.index( first )
    i_last      = list_res.index( last )
    list_res    = list_res[ i_first : i_last+1 ]

    for f in list_res:
        fname   = os.path.join( res, f, log )
        if not os.path.isfile( fname ):
            print( f"{f}  is not a file" )
            continue
        with open( fname, 'r' ) as fd:
            lines   = fd.readlines()
        if not len( lines ):
            print( f"file {f}  has no lines" )
            continue
        string      = get_info ( lines )
        if string is None:
            print( f"{f}  no info found" )
        else:
            print( string )
