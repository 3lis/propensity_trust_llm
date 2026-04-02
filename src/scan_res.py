"""
#####################################################################################################################

    trust propensity project - 2025

    scan current results

#####################################################################################################################
"""

import  os
import  sys

res         = "../res"
log         = "runs.log"

def get_info ( lines ):
    """
    retrieve essential info
    """

    conf        = lines[ : 50 ]     # the configuration part
    model       = "---"
    scenario    = "---"
    augm        = "---"
    easiness    = "0"
    found       = False
    for l in conf:
        if l.startswith( "model_short" ):
            model       = l.split()[ -1 ]
            found       = True
        if l.startswith( "scenario" ):
            scenario    = l.split()[ -1 ].replace( "scenario_", '' )
        if "trustq.py" in l:
            scenario    = "Frazier"
        if l.startswith( "easiness" ):
            easiness    = l.split()[ -1 ]
        if l.startswith( "augmentation" ):
            augm        = l.split()[ -1 ]

    if not found:
        return None

    return model, scenario, augm, easiness


# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================

if __name__ == '__main__':
    list_res    = sorted( os.listdir( res ) )

    if len( sys.argv ) > 1:             # if there is an argument, use it at the latest result to show
        last_res    = sys.argv[ 1 ]
        if last_res in list_res:
            idx         = list_res.index( last_res )
            list_res    = list_res[ idx : ]

    print( "\n________________________________________________________\n" )
    print( "       result       model     scen       augm     ease" )
    print( "_________________________________________________________" )
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
        info        = get_info ( lines )
        if info is None:
            print( f"{f}  no info found" )
        else:
            m, s, a, e        = info
            print( f"{f}  {m:<10} {s:<8} {a:<11} {e}" )
    print( "________________________________________________________\n" )
