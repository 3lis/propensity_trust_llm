"""
#####################################################################################################################

    HURST project - 2024

    Comparison between runs

    comparison can be made between two, three, or many runs
    when the runs to compare are two or three, there will separate plots for the groups
    otherwise only the overall results are compared

    you should provide a list of runs to compare, and optionally a list of labels
    with same length as runs, and an overall title in suptitle

#####################################################################################################################
"""


import  os
import  sys
import  platform
import  time
import  datetime
import  shutil

import  plot

# this is the main choice of plotting:
#   "mx2"   organize data for plotting comparison of an arbitrary number multiruns results, with a desgin space
#           of mx2 independent variables (where 2 is typically for two scenarios)
#   "final" plotting comparison in the format for publication with several separate graphs with same number of bars
#   "comp"  organize data for plotting comparison of an arbitrary number multiruns results
#           just one plot is produced for the overall outcome
#   "lcomp" legacy "comp", an earlier version using boxes, now obsoleted
#   "his2"  produce a history plot with the comparison of two runs
#   "mhis2" generate mutiple comparisons of the kind with "his2", in this case plot_base_name should be a list
plot_type       = "comp"

# this is the main prefix for the plot names, and can be personalized in the specification of groups and comparisons
plot_base_name  = 'plot_ppo'

runs            = [
        "25-01-06_02-18-58",	# zs			0.56  0.24 0.56
#       "25-06-22_11-03-52",	# tom			0.74  0.33 0.75
        "24-12-29_15-20-41",	# ll3ft*		0.65  0.31 0.66
        "24-12-29_16-39-38",	# 24-12-28_18-46-11	0.60  0.28 0.61 DPO
        "24-12-31_11-56-51",	# 24-12-30_19-20-43	0.62  0.26 0.62 ORPO
        "25-01-15_15-37-50",	# 25-01-15_08-27-01	0.58  0.25 0.59 PPO 6epoch
        "25-01-16_08-40-54",	# 25-01-15_19-33-37	0.59  0.27 0.59 PPO 12epoch
        "25-06-21_07-19-15",	# 25-06-19_18-53-37	0.64  0.29 0.64 ll3poa
        "25-06-22_07-58-29",	# 25-06-21_15-25-53	0.64  0.28 0.64 ll3poar
        "25-06-21_10-13-25",	# 25-06-20_17-42-09	0.64  0.27 0.64 ll3pos
        "25-06-23_08-45-42",	# 25-06-17_17-54-18	0.67  0.29 0.67 ll3po5
        "25-06-23_14-51-54",	# 25-06-23_11-25-02	0.65  0.31 0.66 ll3pe0
#       "25-06-22_17-24-33",	# 25-06-19_18-53-37(tm)	0.71  0.29 0.72 ll3poa(tom)
]
labels          = [
    "zero-shot",
#   "2-mem",
    "SFT",
    "ORPO",
    "DPO",
    "PPO offline 6-epochs",
    "PPO offline 12-epochs",
    "PPO online 2000 steps",
    "PPO online ref_model",
    "PPO online rwd success",
    "PPO online 500 steps",
    "PPO online difficulty 3",
#   "PPO online + 2-mem",
]
y_range         = ( 0.4, 0.8 )
y_range_acc     = ( 0.1, 0.4 )


suptitle                = "fine-tuning comparison for Llama.3-1"
twin_plots              = 1
h_legend                = False            # set the legend to be displayed horizontally
by_chance               = True             # include the line of by-chance success
col_idx                 = None             # specific list of indeces for colors
col_idx                 = [
    33,  0,  9,  19,  4,  5, 17, 28, 29, 30, 2,
]

n_runs                  = len( runs )


dir_res             = '../res'
# NOTE the following variables will be validated in init_dirs()
dir_plot            = 'plot'
plot_name           = None
plot_name_noor      = None
plot_name_acc       = None
log_run_name        = 'runs.log'
log_runs            = []
dump_file_name      = 'results.pickle'
dump_files          = []
final_plotdir       = "../plots"            # common directory for final type plots
frmt_plotdir        = "%y-%m-%d-%H-%M"      # datetime format for final type plot directory

# execution directives
DO_NOTHING              = False             # for interactive use


def init_dirs():
    """
    set paths to directories of the runs to compare, and where to save the comparison
    """
    global dir_plot
    global runs, log_runs, dump_files, plot_name, plot_name_acc, plot_name_noor

    if n_runs == 0:                         # when runs are not specified, use group_runs
        # first do a check of group consistency
        leng    = [ len( g ) for g in group_runs ]
        assert len( list( set( leng ) ) ) == 1, \
            f"error: group of results are not of same size: {group_runs}"
        assert len( group_runs ) == len( group_titles ), \
            "error: group of results inconsistent with group titles"
        assert leng[ 0 ] == len( labels ), \
            "error: group of results inconsistent with labels"
        # then flatten all result dirnames from group_runs into runs
        runs    = [ r for group in group_runs for r in group ]
    for r in runs:
        if not len( r ):                    # manage missing runs in a plot group
            dump_files.append( r )
            continue
        dr          = os.path.join( dir_res, r )
        assert os.path.isdir( dr ), f"error in init_dirs, run {dr} not found"
        log_runs.append( os.path.join( dr, log_run_name ) )
        dump_files.append( os.path.join( dr, dump_file_name ) )
    now_time    = time.strftime( frmt_plotdir )
    dir_plot    = os.path.join( final_plotdir, now_time )
    if not os.path.isdir( dir_plot ):
        os.makedirs( dir_plot )

    shutil.copy( "compare.py", dir_plot )   # leave trace of how plots have been generated
    shutil.copy( "plot.py", dir_plot )

    if isinstance( plot_base_name, list ):
        plot_name   = [ os.path.join( dir_plot, p + '_' ) for p in plot_base_name ]
    else:
        plot_name       = plot_base_name + '_'
        plot_name_noor  = plot_base_name + '_noor_'
        plot_name_acc   = plot_base_name + '_acc_'
        plot_name       = os.path.join( dir_plot, plot_base_name + '_' )
        plot_name_acc   = os.path.join( dir_plot, plot_base_name + '_acc_' )
        plot_name_noor  = os.path.join( dir_plot,  plot_base_name + '_noor_' )


def get_runs():
    """
    gather all information and data necessary for the comparison

    return:
        [tuple] with three results structures, the last one is None for two comparisons
    """
    global labels

    results = []
    for d in dump_files:
        if not len( d ):                    # manage missing runs in a plot group
            results.append( None )
            continue
        _, res      = plot.retrieve_pickle( d )
        results.append( res )

    if labels is None:
        labels      = [ r[ -5 : ] for r in runs ]
    if n_runs > 0:
        return results

    # when runs are not specified, reorganize results in groups
    grp_len         = len( group_runs[ 0 ] )
    results_by_grp  = [ results[ i : i + grp_len ] for i in range( 0, len( results ), grp_len ) ]
    return results_by_grp


def do_plots( results ):
    """
    generate color plots of comparisons
    """
    global  twin_plots
    global  y_range
    match plot_type:
        case "final":
            plot.plot_final_comp(
                    results,
                    labels          = labels,
                    group_titles    = group_titles,
                    suptitle        = suptitle,
                    basename        = plot_name,
                    y_range         = y_range,
                    col_idx         = col_idx,
                    twin_plots      = twin_plots,
                    by_chance       = by_chance,
                    h_legend        = h_legend
            )
            if y_range_acc is not None:
                y_range             = y_range_acc
            plot.plot_final_comp(
                    results,
                    labels          = labels,
                    group_titles    = group_titles,
                    suptitle        = suptitle,
                    basename        = plot_name_acc,
                    accuracy        = True,
                    y_range         = y_range,
                    col_idx         = col_idx,
                    twin_plots      = twin_plots,
                    by_chance       = by_chance,
                    h_legend        = h_legend
            )
        case "mx2":
            plot.plot_comp_mx2( results, m2, labels=labels, xlabels=xlabels, basename=plot_name )
        case "comp":
            plot.plot_comp(
                    results,
                    labels          = labels,
                    suptitle        = suptitle,
                    basename        = plot_name,
                    accuracy        = False,
                    y_range         = y_range,
                    col_idx         = col_idx,
                    by_chance       = by_chance,
                    h_legend        = h_legend
            )
            if y_range_acc is not None:
                y_range             = y_range_acc
            plot.plot_comp(
                    results,
                    labels          = labels,
                    suptitle        = suptitle,
                    basename        = plot_name_acc,
                    accuracy        = True,
                    y_range         = y_range,
                    col_idx         = col_idx,
                    by_chance       = by_chance,
                    h_legend        = h_legend
            )
        case "lcomp":   # legacy com
            plot.plot_comparison( results, labels=labels, basename=plot_name, suptitle=suptitle, twin_plots=twin_plots )
            plot.plot_comparison( results, labels=labels, basename=plot_name_noor, suptitle=suptitle, no_oracle=True )
            plot.plot_acc_comparison( results, labels=labels, basename=plot_name_acc, suptitle=suptitle, twin_plots=twin_plots )
        case "his2":
            plot.plot_two_success_history( dump_files, labels=labels, basename=plot_name )
            plot.plot_two_accuracy_history( dump_files, labels=labels, basename=plot_name, which_acc="faccuracy" )
            plot.plot_two_accuracy_history( dump_files, labels=labels, basename=plot_name+'old_', which_acc="accuracy")
        case "mhis2":
            iters       = range( 0, len( dump_files ), 2 )
            dump_files2 = [ [ dump_files[ i ], dump_files[ i+1 ] ] for i in iters ]
            for d, l, p in zip( dump_files2, labels, plot_name ):
                plot.plot_two_success_history( d, labels=l, basename=p )
                plot.plot_two_accuracy_history( d, labels=l, basename=p, which_acc="faccuracy" )
                plot.plot_two_accuracy_history( d, labels=l, basename=p+'old_', which_acc="accuracy")

# ===================================================================================================================
#
#   MAIN
#
# ===================================================================================================================
if __name__ == '__main__':

    init_dirs()
    results     = get_runs()
    if not DO_NOTHING:
        do_plots( results )
    else:
        print( "\nprogram set for doing nothing\n" )
