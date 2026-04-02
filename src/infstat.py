"""
#####################################################################################################################

    Program for generating statistics over a range of executions in ../res

    the overall choice of statistics analysis is with the variable analysis, and the
    executions to analyze are specified in res_range

    for textual statistics the detailed specifications for which analyses to perform are in head of do_stat()

    NOTE: lines commented and marked with # TMP have been used to print out the numerical values of plots

#####################################################################################################################
"""

import  os
import  sys
import  re
import  json
import  time
import  shutil
import  numpy   as np
import  pandas  as pd
import  plot
from    scipy.stats                 import pearsonr
from    statsmodels.formula.api     import ols, mixedlm
from    statsmodels.stats.anova     import anova_lm
from    itertools                   import chain
from    models                      import models_short_name

DO_NOTHING          = False                 # for interactive use
DO_PLOTS            = True                  # generate plots
DO_STATS            = False                 # generate statistics

# specification of the executions to analyze
# if res_range is empty all executions found in ../res are analyzed
# if res_range has only one entry, it is the first execution to process, followed by all the others
# if res_range has two entries, these are the boundaries of the executions to analyze
# if res_range has one list, than all and only the entries in the inner list are analyzed
# if res_range has one or more tuples, than tuples should have two entries, which are boundaries of multiple ranges
#res_range           = [ "25-09-01_10-54-57", "25-09-02_22-06-08" ]  # 10x50 no_trust/zero_shot/tom fire
#res_range           = [ "25-09-01_10-54-57", "25-09-04_13-11-01" ]  # 10x50 all
res_range           = [ "25-09-02_05-06-23", "25-09-04_13-11-01" ]  # 10x50 tom fire/farm/school

analysis            = "general"
analysis            = "augmentations"
analysis            = "scenarios"

res                 = "../res"                  # results directory
dir_json            = "../data"                 # directory with all input data
dir_stat            = "../stat"                 # output directory

dump_file           = "df.pkl"                  # filename of results in pandas format
f_bstat             = "bstat.txt"               # filename of output basic statistics
f_tstat             = "tstat.txt"               # filename of output statistics for single task
f_hstat             = "hstat.txt"               # filename of output higher statistics
f_ltabs             = "tables.tex"              # filename of LaTeX output tables
f_plot              = "pl_"                     # filename prefix of output plots
frmt_statdir        = "%y-%m-%d-%H-%M"          # datetime format for output directory


# the models actually used so far, in a convenient order
model_list  = [
        "gpt35",
        "gptoss",
        "gpt4om",
        "gpt4",
        "gpt4o",
        "gpt41m",
        "ll2-7",
        "ll2-13",
        "ll3-8",
        "ph3m",
        "gem2-9",
        "qw1-7",
        "qw2-7",
        "qw2-14",
        "cl3h",
        "cl3.5h",
        "cl3.5s",
        "cl3.7s",
        "cl3o",
]
# indeces into model_list for plot grouping
model_idx   = { "openai": ( 0, 6 ), "meta_phi": ( 6, 10 ), "qwen_gem": ( 10, 14 ), "anthro": ( 14, 19 ) }
 
# augmentations used so far, in a convenient order
augm_list   = [ "no_trust", "zero_shot", "tom" ]
augm_plist  = [ "no_trust", "zero_shot", "2-mem" ]      # version for plot labels
 
# possible outcome
outc_list   = [ 4, 5, 6, 8, 9, 10 ]

# scenarios, in a convenient order
scen_list   = [ "fire", "farm", "school" ]


def select_data():
    """
    build the list of results to collect for statistics

    return:             [list] with directories in ../res
    """
    list_res    = sorted( os.listdir( res ) )
    if not len( res_range ):
        return list_res

    if isinstance( res_range[ 0 ], list ):
        return res_range[ 0 ]

    if isinstance( res_range[ 0 ], tuple ):
        multi_res   = []
        for r in res_range:
            assert len( r ) == 2, "entries in res_range should be tuples with exactely two items"
            first   = r[ 0 ]
            last    = r[ -1 ]
            assert first in list_res, f"first specified result {first} not found"
            assert last in list_res, f"last specified result {last} not found"
            i_first     = list_res.index( first )
            i_last      = list_res.index( last )
            multi_res   += list_res[ i_first : i_last+1 ]
        return multi_res

    if len( res_range ) == 1:
        first   = res_range[ 0 ]
        assert first in list_res, f"first specified result {first} not found"
        i_first     = list_res.index( first )
        return list_res[ i_first : ]

    if len( res_range ) == 2:
        first   = res_range[ 0 ]
        last    = res_range[ -1 ]
        assert first in list_res, f"first specified result {first} not found"
        assert last in list_res, f"last specified result {last} not found"
        i_first     = list_res.index( first )
        i_last      = list_res.index( last )
        return list_res[ i_first : i_last+1 ]

    print( "if you want to specify single results to be collected, include them in a list inside res_range\n" )
    return []




def collect_data():
    """
    Scan the results, collecting all data

    return:             [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """

    list_res    = select_data()
    n_res       = len( list_res )
    print( f"scanning for {n_res} execution results\n" )
    dfs         = []

    for f in list_res:                          # scan all selected results
        fname   = os.path.join( res, f, dump_file )
        if not os.path.isfile( fname ):
            print( f"{f}  is not a file" )
            continue
        df      = pd.read_pickle( fname )
        n_rec   = len( df )
        dfs.append( df )
        print( f"{f}  done with {n_rec} records" )


    df      = pd.concat( dfs )

    df[ "model" ]   = pd.Categorical( df[ "model" ], categories=model_list, ordered=True )
    df[ "augm" ]    = pd.Categorical( df[ "augm" ], categories=augm_list, ordered=True )
    df[ "scen" ]    = pd.Categorical( df[ "scen" ], categories=scen_list, ordered=True )

    return df


def print_means( f, df, groups, scores='decision' ):
    """
    print means and std of the main scores grouped as requested

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        groups          [list] of lists with independent variables column names
        scores          [list] of dependent variables column names

    """

    def se(x):
        """Standard error of the mean"""
        return x.std( ddof=1 ) / np.sqrt( len( x ) )
    
    for group in groups:
        if len( group ):
            res     = df.groupby( group, observed=True )[ scores ].agg( [ "mean", "std", se ] ).round( 3 )
        else:
            res     = df[ scores ].agg( [ "mean", "std", se ] ).round( 3 )
        f.write( f" {group} ".center( 80, '=' ) + '\n\n' )
        f.write( res.to_string() + '\n\n' )
        f.write( 80 * "=" + "\n\n" )

def normalize( x ):
    """
    normalize in range 0-1
    """
    if isinstance( x, list ):       # first, ensure x is already a numpy array, otherwise convert it
        x   = np.array( x )
    lo, hi  = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)



def get_means( df, group=None, scores="decision", output="numpy" ):
    """
    get means and se of the main scores grouped as requested

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        group           [strt] column name
        scores          [list] of dependent variables column names
        output          [str] output format: "numpy" | "dict"

    return:
        in the case of output="numpy" an array with two rows, first for mean and second for se, and rows
        corresponding to the values of the group variable
        in the case of output="dict" a dictionary with main keys the values of the group variable, and values
        another dict with keys "mean" and "se"

    """

    def se(x):
        """Standard error of the mean"""
        return x.std( ddof=1 ) / np.sqrt( len( x ) )
    
    if group is None:
        res     = df[ scores ].agg( [ "mean", se ] )
    else:
        res     = df.groupby( group, observed=True )[ scores ].agg( [ "mean", se ] )

    match output:
        case "numpy":
            return res.to_numpy().T
        case "dict":
            return res.transpose().to_dict()


def latex_scenario( f, df, w_std=0.5, w_eta=0.5 ):
    """
    print several columns related to the dependency of trust from scenarios
    rows are sorted so that the top models are those with lowest ICC

    params:
        f               [_io.TextIOWrapper]
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """
    small_open  = r"{\small$\pm"
    table_open  = r"\begin{tabular}{"
    close_small = r"$}"
    close       = "}"
    close_table = r"\end{tabular}"
    eol         = r"\\"

    score       = "decision"
    group       = "scen"

    f.write( "%% special statistics for P2T" )
    f.write( "\n%\n%\n" )

    res_dict    = dict()

    all_std     = list()
    all_eta     = list()
    for m in model_list:
        dfm             = df[ df[ "model" ] == m ].copy()
        icc             = get_icc( dfm )
        eta2            = get_eta_squared( dfm )
        all_eta.append( eta2 )
        means           = dfm.groupby( group, observed=True )[ score ].mean()
        overall         = means.mean()
        gr_range        = means.max() - means.min()
        std             = means.std()
        all_std.append( std )
        res_dict[ m ]   = ( overall, gr_range, std, eta2, icc )

    std_n               = normalize( all_std )
    eta_n               = normalize( all_eta )
    p2ti                = w_std * ( 1 - std_n ) + w_eta * ( 1 - eta_n )
    for i, m in enumerate( model_list ):
        res_dict[ m ]   = res_dict[ m ] + ( p2ti[ i ], )

    sorted_keys         = sorted( res_dict, key=lambda k: res_dict[k][ -1 ], reverse=True )

    heads       = ( "model", "overall mean", "scenario range", "scenario std", r"$\eta^2$", "ICC", "P2TI" )
    nc          = len( heads )
    f.write( table_open + 'l' + (nc-1) * 'c' + close + '\n' )
    f.write( r"\toprule" + '\n' )
    f.write( f"{heads[0]}\t" )
    for h in heads[ 1: ]:
        f.write( f"& {h}\t" )
    f.write( f"{eol}\n" )
    f.write( r"\midrule" + '\n' )
    f.write( "%\n" )
    for m in sorted_keys:
        f.write( f"{m:<10}" )
        results     = res_dict[ m ]
        for i, r in enumerate( results ):
            match i:
                case 0 | 5:
                    f.write( f"\t& {r:5.2f} " )
                case 3:
                    f.write( f"\t& {r:6.4f} " )
                case _:
                    f.write( f"\t& {r:6.3f} " )
        f.write( f"{eol}\n" )
    f.write( r"\bottomrule" + '\n' )
    f.write( close_table )
    f.write( "\n%\n%\n" )


def latex_means_all( f, df ):
    """
    print means and std of the main score as LaTeX tabular, organized by models
    rows are sorted so that the top models are those with highest mean in the "all" category

    params:
        f               [_io.TextIOWrapper]
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """
    small_open  = r"{\small$\pm"
    table_open  = r"\begin{tabular}{"
    close_small = r"$}"
    close       = "}"
    close_table = r"\end{tabular}"
    eol         = r"\\"

    score       = "decision"
    groups      = [ "augm", "scen" ]

    f.write( "%% basic statistics for groups " )
    for group in groups:
        f.write( f" {group} " )
    f.write( "\n%\n%\n" )

    res_dict    = dict()

    for m in model_list:
        dfm             = df[ df[ "model" ] == m ].copy()
        res             = get_means( dfm, output="dict" )
        res_dict[ m ]   = { "all": res }
        for g in groups:
            if g == "scen":     # to make scenario comparable, ensure to use tom augomentation only
                dfm             = dfm[ dfm[ "augm" ] == "tom" ].copy()
            res             = get_means( dfm, group=g, output="dict" )
            res_dict[ m ].update( res )

    sorted_keys         = sorted( res_dict, key=lambda k: res_dict[k]['all']['mean'], reverse=True )

    cats        = list()
    for g in groups:
        cats    += df[ g ].unique().tolist()
    cats        += [ "all" ]
    nc          = len( cats )
    f.write( table_open + 'l' + nc * 'c' + close + '\n' )
    f.write( r"\toprule" + '\n' )
    for c in cats:
        f.write( f"& {c}\t" )
    f.write( f"{eol}\n" )
    f.write( r"\midrule" + '\n' )
    f.write( "%\n" )
    for m in sorted_keys:
        f.write( f"{m:<10}" )
        resm    = res_dict[ m ]
        for c in cats:
            mean        = res_dict[ m ][ c ][ "mean" ]
            se          = res_dict[ m ][ c ][ "se" ]
            f.write( f"\t& {mean:5.2f}({small_open}{se:3.2f}{close_small}) " )
        f.write( f"{eol}\n" )
    f.write( r"\bottomrule" + '\n' )
    f.write( close_table )
    f.write( "\n%\n%\n" )


def pearson( df, x, y ):
    """
    compute Pearson's coefficient
    if both x and y are numeric, use the quick pearsonr, otherwise ols
    in the latter case, construct the regression formula in the 'R'-style language
    the p value derives from Student t applied to
        t = r * sqrt( ( n-2 ) / ( 1 - r^2 ) )

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        x               [str] one of the independent variables
        y               [str] one of the dependent variables

    return:             [tuple] r, p-value
    """
    numeric     = df[ x ].dtype.kind in 'biuf'  # bool/int/uint/float
    if numeric:
        r, p    = pearsonr( df[ x ], df[ y ] )
        return r, p

    formula     = f"{y} ~ C({x})"
    model       = ols( formula, df ).fit()
    r2          = model.rsquared
    r2          = max( r2, 0.0 )                # for tiny numbers may be even negative
    r           = np.sqrt( r2 )
    p           = model.pvalues.values[ 1 ]
    return r, p


def anova( df, x, y ):
    """
    compute one-way anova of one independent categorial variable x against the result y
    Construct the regression formula in the 'R'-style used in the ols function of statsmodels,
    then apply the anova_lm to the regression model returned by ols
    See:
        https://www.statsmodels.org/dev/generated/statsmodels.formula.api.ols.html
        https://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
    Moreover, compute the effect size by eta^2 of F, then squared, assuming that degree-of-freedom_Hyp=1
        r = sqrt( (F * df1 ) / ( F * df1 + df2 ) )

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
        x               [str] one of the independent variables
        y               [str] one of the dependent variables

    return:             [tuple] F, df2, r, p-value
    """
    n           = len( df ) - 2
    formula     = f"{y} ~ C({x})"
    model       = ols( formula, df ).fit()
    a           = anova_lm( model )
    va          = a.values
#   a.values is orgamized as:
#   [
#       [ df sum_sq mean_sq F PR(>F) ]      # for C({x})
#       [ df sum_sq mean_sq F PR(>F) ]      # for Residual
#   ]
#
#   print( va )
    f           = va[ 0 ][ 3 ]                      # this is F
    p           = va[ 0 ][ 4 ]                      # this is p-value
    sum_sq_x    = va[ 0 ][ 1 ]                      # sum_sq for C({x})
    sum_sq_res  = va[ 1 ][ 1 ]                      # sum_sq for Residual

    ss_effect   = sum_sq_x
    ss_total    = sum_sq_x + sum_sq_res
    eta2        = ss_effect / ss_total

    return f, p, eta2


def get_icc( df, group="scen", score="decision" ):
    """
    return the intra-class correlation coefficient (ICC)
    arg:
        df      [pandas.core.frame.DataFrame]
        group   [str] the group for which ICC is computed
        score   [str] which dependent variable to use
    """

    df[ "score" ]   = df[ score ].astype( int )
    model           = mixedlm("score ~ 1", df, groups=group )
    fit             = model.fit()
    var_group       = fit.cov_re.iloc[0,0]      # group variance
    var_resid       = fit.scale                 # residual variance
    return var_group / ( var_group + var_resid )

def get_eta_squared( df, group="scen", score="decision" ):
    """
    return eta squared for scenario effect
    arg:
        df      [pandas.core.frame.DataFrame]
        group   [str] the group for which ICC is computed
    """
    df[ "score" ]   = df[ score ].astype( int )
    model           = ols( f"score ~ C({group})", data=df ).fit()
    anova           = anova_lm( model, typ=2 )
    ss_group        = anova.loc[ f"C({group})", "sum_sq" ]
    ss_total        = anova[ "sum_sq" ].sum()
    return ss_group / ss_total

def mean_se( df, group="outcome", score="decision" ):
    """
    return the mean and the correct standard error cmulated over all elements of group, and separated
    arg:
        df      [pandas.core.frame.DataFrame]
        group   [str] the group for which compsite and separated stastistics are requested
        score   [str] which dependent variable to use
    """
    res_dict    = dict()
    agg         = [ "mean", "std", "count" ]
    outs        = [ 4, 5, 6, 8, 9, 10 ]         # the possible outcome values for agent_outcome_float()

    # first compute cumulative mean, std_err, and count
    by_group        = df.groupby( group, observed=True )[ score ].agg( agg )
    stats           = by_group.to_dict()
    for o in outs:
        mean            = stats[ "mean" ][ o ]
        std             = stats[ "std" ][ o ]
        sqrt_count      = np.sqrt( stats[ "count" ][ o ] )
        se              = std / sqrt_count
        res_dict[ o ]   = { "mean": mean, "std":std, "se": se }

    return res_dict


def print_anova( f, df, group ):
    """
    Print main anova results

    params:
        df              [pandas.core.frame.DataFrame] the data in pandas DataFrame
    """
    f.write( 80 * "=" + "\n" )
    f.write( "ANOVA analysis of variable against scores ".center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( f"model    variable        F     eta^2  p-value\n" )
    f.write( "_______________________________________________\n" )
    m               = "all"
    # cycle over all groups
    for x in group:
        F, p, e     = anova( df, x, "scores" )
        f.write( f"{m:<8} {x:<10}{F:8.3f}  {e:5.4f}   {p:6.5f}\n" )
    models      = df.model.unique().tolist()
    models.sort()
    # do the same as above, separated by models
    if len( models ) > 1:
        for m in models:
            dfm     = df[ (df[ 'model' ]==m ) ]
            for x in group:
                F, p, e     = anova( dfm, x, "scores" )
                f.write( f"{m:<8} {x:<10}{F:8.3f}  {e:5.4f}   {p:6.5f}\n" )
    f.write( "_________________________________________________\n\n" )


def do_stat( df ):
    """
    do all statistics and write it on file
    which statistics, both at basic and high level, are set with lists detailing
    the group of columns to be analyzed

    """
    title       = f"{analysis} analysis"
    match analysis:
        case "augmentations":
            all_groups  = [
                [],
                [ "model" ],
                [ "family" ],
                [ "outcome" ],
                [ "augm" ],
                [ "augm", "outcome" ],
                [ "model", "augm" ],
                [ "family", "augm" ],
            ]
            # this group includes the additional "level" column, that split outcome into two levels
            spl_groups  = [
                [ "model", "level" ],
                [ "family", "level" ],
                [ "augm", "level" ],
            ]
            corr_group  = []
            anova_group = []
        case "scenarios":
            all_groups  = [
                [],
                [ "model" ],
                [ "family" ],
                [ "outcome" ],
                [ "scen" ],
                [ "scen", "outcome" ],
                [ "model", "scen" ],
                [ "family", "scen" ],
            ]
            # this group includes the additional "level" column, that split outcome into two levels
            spl_groups  = [
                [ "model", "level" ],
                [ "family", "level" ],
                [ "scen", "level" ],
            ]
            corr_group  = []
            anova_group = []
            # produce also the scenario- related P2T analysis
            fname       = os.path.join( dir_stat, f_ltabs )
            f           = open( fname, 'w' )
            f.write( "% LaTeX tables\n%\n%\n" )
            latex_scenario( f, df )
            f.close()
 
        # in this case just the overall mean table is produced, directly in LaTeX
        case "general":
            fname       = os.path.join( dir_stat, f_ltabs )
            f           = open( fname, 'w' )
            f.write( "% LaTeX tables\n%\n%\n" )
            latex_means_all( f, df )
            f.close()
            return True

    # do basic statistics first
    fname       = os.path.join( dir_stat, f_bstat )
    f           = open( fname, 'w' )
    f.write( 80 * "=" + "\n\n" )
    f.write( "basic statistics".center( 80, ' ' ) + '\n\n' )
    f.write( title.center( 80, ' ' ) + '\n' )
    f.write( 80 * "=" + "\n\n" )
    f.write( 80 * "+" + "\n\n" )
    f.write( "overall decision rates".center( 80, ' ' ) + '\n\n' )
    f.write( 80 * "+" + "\n" )
    print_means( f, df, all_groups, scores="decision" )

    if len( spl_groups ):
        df[ "level" ]   = pd.Categorical(
            np.where( df[ "outcome" ] > 7, "high", "low" ),
            categories=[ "low", "high" ],
            ordered=True
        )
        f.write( 80 * "+" + "\n\n" )
        f.write( "split decision rates".center( 80, ' ' ) + '\n' )
        f.write( 80 * "+" + "\n\n" )
        print_means( f, df, spl_groups, scores="decision" )

    f.close()

    if ( not len( corr_group ) ) and ( not len( anova_group ) ):
        return True

    # now do higher statistics
    fname       = os.path.join( dir_stat, f_hstat )
    f           = open( fname, 'w' )
    f.write( 80 * "+" + "\n" )
    f.write( "higher statistics".center( 80, ' ' ) + '\n' )
    f.write( title.center( 80, ' ' ) + '\n' )
    f.write( 80 * "+" + "\n\n" )
    print_type_corr( f, df )
    f.write( 80 * "+" + "\n\n" )
#   print_anova( f, df, anova_group )

    f.close()

    return True


def eval_success( df ):
    """
    evaluate the "success" column in a proper way, taking into account cases where the model has
    looped over all agents without deciding for one of them
    """
    agents_per_loop = 6

    df = df.copy()

# 1. Identify contiguous blocks of the same task
#    (if you have a simulation id, include it here too: ['sim', 'task'])
    df['task_block'] = (df['task'] != df['task'].shift()).cumsum()

# 2. Within each block, number the rows
    df['idx'] = df.groupby('task_block').cumcount()

# 3. Each cycle is a chunk of 6 rows
    df['cycle_id'] = df['idx'] // agents_per_loop

# 4. Mark last row in each cycle (i.e. last agent in the loop)
    is_last_in_cycle = (df['idx'] % agents_per_loop) == (agents_per_loop - 1)

# 5. For each block+cycle, check if there was any True decision
    dec_any = df.groupby(['task_block', 'cycle_id'])['decision'].transform('any')

# 6. A "loop failure" = last row of cycle AND no decision in that cycle
    df['loop_fail'] = is_last_in_cycle & (~dec_any)

# 7. to_cont is True if either
#    - we actually took a decision, or
#    - we completed a loop with no decision (loop_fail)
    df['to_cont'] = df['decision'] | df['loop_fail']
    
    return df[ df[ 'to_cont' ] ]


def do_line_plot( df, group="augm", basename="pl_" ):
    """
    do one multi-lines plot
    """
    match group:
        case "augm":
            xlabels         = augm_plist
            y_range         = ( 0.0, 0.8 )
        case "outcome":
            xlabels         = outc_list
            y_range         = ( 0.0, 1.0 )
        case _:
            print( f"invalid group {group} in do_line_plots" )
            sys.exit()

    n               = len( model_list )
    all_colors      = plot.gen_colors( n, 3, min_value=0.4, saturation=0.9 )
    use_colors      = all_colors[ 1 ]               # the intermediate brighter
    ylabel          = True                          # draw ylabel for teh first plot only
    for k in model_idx.keys():  
        i0, i1      = model_idx[ k ]
        m_list      = model_list[ i0 : i1 ]
        colors      = use_colors[ i0 : i1 ]
        res_dict    = dict()
        for m in m_list:
            dfm             = df[ df[ "model" ] == m ].copy()
            res_dict[ m ]   = get_means( dfm, group )
#       print( res_dict )       # TMP
#       continue                # TMP
        plot.plot_lines(
                res_dict,
                m_list,
                xlabels,
                colors      = colors,
                ylabel      = ylabel,
                y_range     = y_range,
                suptitle    = f"{k}",
                basename    = f"{basename}{group}_{k}" )
        ylabel      = False


def do_line_plots( df ):
    """
    do all multi-lines plots
    """
    match analysis:
        case "augmentations":
            basename        = os.path.join( dir_stat, f_plot )
            do_line_plot( df, group="augm", basename=basename )
            assert len( df.scen.unique().tolist() ) == 1, "too many scenarios for augmentations analysis"
            assert df.scen.unique().tolist()[ 0 ] == "fire", "wrong scenario for augmentations analysis"
            for a in augm_list:
#               print( f"\n =========== {a} ============" )         # TMP
                basename    = os.path.join( dir_stat, f_plot + a + '_' )
                dfa         = df[ df[ "augm" ] == a ].copy()
                do_line_plot( dfa, group="outcome", basename=basename )


def do_bar_plot( df, basename, suptitle="", ylabel="", scores="decision", y_range=None ):
    """
    do one bar plot
    """
    group           = "model"
    mes         = []
    for s in scen_list:
        dfs             = df[ df[ "scen" ] == s ].copy()
        me              = get_means( dfs, group, scores=scores )
        mes.append( me )
    lists       = [ list( zip( row[ 0 ], row[ 1 ] ) ) for row in mes ]
    res         = list( chain.from_iterable( zip( *lists ) ) )
#   if scores == "success":             # TMP
#       print( res )                    # TMP
#       sys.exit()                      # TMP
    plot.plot_bars(
            res,
            model_list,
            twin_plots  = 3,
            y_range     = y_range,
            suptitle    = suptitle,
            ylabel      = ylabel,
            basename    = basename )


def do_bar_plots( df ):
    """
    do all bar plots
    """
    match analysis:
        case "scenarios":
            assert len( df.augm.unique().tolist() ) == 1, "too many augmentations for scenarios analysis"
            assert df.augm.unique().tolist()[ 0 ] == "tom", "wrong augmentation for scenarios analysis"
            ylabel      = "fraction of entrustment"
            basename    = os.path.join( dir_stat, f_plot + 'scen' )
            suptitle    = "all models, all scenarios"
            do_bar_plot( df, basename, ylabel=ylabel, suptitle=suptitle )

            dfo         = df[ df[ "outcome" ] < 7 ].copy()
            y_range     = ( 0.0, 0.5 )
            basename    = os.path.join( dir_stat, f_plot + 'scen_low' )
            suptitle    = "all models, all scenarios, lower outcomes"
            do_bar_plot( dfo, basename, suptitle=suptitle, ylabel=ylabel, y_range=y_range )

            dfo         = df[ df[ "outcome" ] > 7 ].copy()
            y_range     = ( 0.0, 1.0 )
            basename    = os.path.join( dir_stat, f_plot + 'scen_high' )
            suptitle    = "all models, all scenarios, higher outcomes"
            do_bar_plot( dfo, basename, suptitle=suptitle, ylabel=ylabel, y_range=y_range )

            ylabel      = "fraction of success"
            dfs         = eval_success( df )
            y_range     = ( 0.2, 1.0 )
            basename    = os.path.join( dir_stat, f_plot + 'scen_succ' )
            suptitle    = "task success for all models, all scenarios"
            do_bar_plot( dfs, basename, suptitle=suptitle, scores="success", ylabel=ylabel, y_range=y_range )



# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================
if __name__ == '__main__':
    if DO_NOTHING:
        print( "program instructed to do nothing" )
    else:
        df          = collect_data()        # all data in pandas DataFrame
        now_time    = time.strftime( frmt_statdir )
        dir_stat    = os.path.join( dir_stat, now_time )
        if not os.path.isdir( dir_stat ):
            os.makedirs( dir_stat )
        # save a copy of this file and the plotting script
        shutil.copy( "infstat.py", dir_stat )
        shutil.copy( "plot.py", dir_stat )
        fname   = os.path.join( dir_stat, dump_file )
        df.to_pickle( fname )
        if DO_PLOTS:
            do_line_plots( df )
            do_bar_plots( df )
        if DO_STATS:
            do_stat( df )
