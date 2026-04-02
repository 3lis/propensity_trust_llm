"""

functions for plotting results

alex Aug 2025

"""

import  os
import  pickle
import  numpy               as np
import  copy
import  colorsys
from    matplotlib          import pyplot
from    matplotlib.patches  import Patch, Polygon
from    matplotlib.lines    import Line2D

bfigsize        = ( 18.0, 6.0 )                         # figure size for bar plots
lfigsize        = ( 12.0, 8.0 )                         # figure size for line plots
rfigsize        = ( 14.0, 8.0 )                         # figure size for radar plot
labelspacing    = 1.1
extension       = ".pdf"

# the elegant tableau-10 series of colors in matplotlib
# NOTE: there are - of course - just 10 colors
tab_colors  = [ 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
          'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan' ]
line_style  = [ '-', '--', '-', '--' ]

xlabel_rot  = 45                                        # X labels rotation, used in multiple runs box plots
char_len    = 0.007                                     # typical length of a character in legend, in plot units


# ===================================================================================================================
#   utilities
#   simple help functions used by several plotting functions
#
#   gen_colors()
#   condifence_interval()
#   pretty_limits()
# ===================================================================================================================

def gen_colors( n_hues, n_levels, min_value=0.2, saturation=0.8 ) :
    """
    Generates a color palette matrix with customizable intensity and saturation.

    params:
        n_hues      [int] number of different hues (columns)
        n_levels    [int] number of intensity levels (rows)
        min_value   [float] minimum brightness (0 < min_value < 1)
        saturation  [float] saturation of colors (0 to 1)

    Returns:
        List of lists of hex color strings.
    """
    if not ( 0 < min_value <= 1 ):
        raise ValueError("min_value should be between 0 and 1.")
    if not ( 0 <= saturation <= 1 ):
        raise ValueError("saturation should be between 0 and 1.")

    palette     = []
    for level in range( n_levels ):
        # Linear interpolation of value between min_value and 1.0
        value   = min_value + ( level / (n_levels - 1) ) * ( 1 - min_value ) if n_levels > 1 else min_value
        row     = []
        for hue_index in range( n_hues ):
            hue = hue_index / n_hues
            r, g, b = colorsys.hsv_to_rgb( hue, saturation, value )
            hex_color = '#{:02x}{:02x}{:02x}'.format( int(r * 255), int(g * 255), int(b * 255) )
            row.append( hex_color )
        palette.append( row )
    return palette


def condifence_interval( data, critical_value=1.96 ):
    """
    compute confidence interval of data (there is no a direct function for that!)

    args:
        data:           [numpy.array]
        critical_value: [float] 1.96 for 95% confidence
    return:
                        [float]
    """
    s       = data.std()
    n       = data.shape[ 0 ]
    c_i     = s * critical_value / np.sqrt( n )
    return  c_i


def pretty_limits( x0, x1, clip=True ):
    """
    set fractionary limits for a scale

    args:
        x0:             [float] lower limit
        x1:             [float] upper limit
    return:
                        [tuple] new limits
    """
    x0  = int( 10 * x0 ) / 10
    if clip:
        x1  = min( 1.0, int( 1 + 10 * x1 ) / 10 )
    else:
        x1  = int( 1 + 10 * x1 ) / 10

    return x0, x1



# ===================================================================================================================
#   plotting functions
#
#   plot_lines()
#   plot_bars()
#   plot_multi_bars()
#   plot_radar()
# ===================================================================================================================


def plot_lines(
        res_dict,
        labels,
        xlabels,
        xmargin     = 0.2,
        suptitle    = '',
        basename    = "plot_",
        h_legend    = False,
        width       = 2,
        colors      = None,
        ylabel      = False,
        y_range     = None ):
    """
    plotting bars of averaged results, optionally grouped

    input:
        res_dict        [dict] of [tuple] with keys as labels, and (mean_list,se_list) as values
        labels          [tuple] strings for the results in each subplots
        xlabels         [list] of [str] with x tics labels
        xmargin         [float] aount of right and left margins for X
        suptitle        [str] superior title of the plot
        basename        [str] basename of the file with the plot
        h_legend        [bool] set the legend to be displayed horizontally
        width           [float] bars width
        colors          [list] with all necessary colors, in any Matplotlib allowed format, None for automatic
        y_range         [tuple] of min and max Y, and optionally Y tick
    """
    n       = len( labels )
    assert n == len( res_dict ), "mismatched results dict and labels"
    nx      = len( xlabels )
    assert nx == len( res_dict[ labels[ 0 ] ][ 0 ] ), "mismatched results dict and xlabels"
    assert nx == len( res_dict[ labels[ 0 ] ][ 1 ] ), "mismatched results dict and xlabels"

    if isinstance( xlabels[ 0 ], int ): # maintains xlabels as x position if values are integers numbers
        x   = xlabels
        rot = 0
    else:                               # otherwise s positions are in steps of 1
        x   = 1 + np.arange( nx )
        rot = xlabel_rot

    if colors is None:
        if n < 10:
            colors  = tab_colors
        else:
            palette = gen_colors( n, 1, min_value=0.6, saturation=0.5 )
            colors  = list( palette[ 0 ] )

    pyplot.rcParams.update( { "font.size": 20 } )
    handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, labels ) ]
    fig, ax     = pyplot.subplots( figsize=lfigsize )

    for i, l in enumerate( labels ):
        y, e    = res_dict[ l ]
        ax.errorbar( x, y, yerr=e, color=colors[ i ], linewidth=width )
    left        = x[ 0 ] - xmargin
    right       = x[ -1 ] + xmargin
    ax.set_xlim( left=left, right=right )
    ax.set_xticks( x, labels=xlabels, rotation=rot )
    if y_range is not None:
        ax.set_ylim( bottom=y_range[ 0 ], top=y_range[ 1 ] )
        if len( y_range ) > 2:
            y_tick  = y_range[ -1 ]
        else:
            y_tick  = 0.1
        ax.set_yticks( np.arange( y_range[ 0 ], y_range[ 1 ]+0.01, y_tick ) )


    if ylabel:
        ax.set_ylabel( "fraction of entrustment" )

    # in the case of legend in horizonal, use one or more rows depending on the number of labels,
    # and align with the first group if one rows, otherwise with the second group
    if h_legend:
        ncol        = n // 2
        x_bbox      = 0.0
        y_bbox      = -0.4
        ax.legend(
                handles         = handles,
                loc             = "lower left",
                bbox_to_anchor  = [ x_bbox, y_bbox ],
                ncol            = ncol,
                labelspacing    = labelspacing
        )
    else:
        ax.legend( handles=handles, loc="center", bbox_to_anchor=[0.15,0.75], labelspacing=labelspacing )
    fname       = basename + extension
    pyplot.margins( x=0 )
    pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()



def plot_bars(
        res,
        xlabels,
        twin_plots  = 3,
        suptitle    = '',
        ylabel      = '',
        basename    = "plot_",
        h_legend    = False,
        width       = 1,
        colors      = None,
        y_range     = None ):
    """
    plotting bars of averaged results, optionally grouped

    input:
        res             [list] of [tuple] with (mean_list,se_list) as values
        xlabels         [list] of [str] with x tics labels
        xmargin         [float] aount of right and left margins for X
        twin_plots      [int] plots should be grouped in twin_plots, 1 for no grouping
        suptitle        [str] superior title of the plot
        ylabel          [str] Y label
        basename        [str] basename of the file with the plot
        h_legend        [bool] set the legend to be displayed horizontally
        width           [float] bars width
        colors          [list] with all necessary colors, in any Matplotlib allowed format, None for automatic
        y_range         [tuple] of min and max Y, and optionally Y tick
    """
    nx      = len( xlabels )
    n       = nx * twin_plots
    nr      = len( res )
    assert  n == nr, f"mismatched results list {nr} and xlabels {nx} with twin_plots {twin_plots}"
    x       = [  i + i // twin_plots for i in range( n ) ]
    xpos    = x[ ::twin_plots ]

    if colors is None:
        if twin_plots > 1:
            palette = gen_colors( nx, twin_plots, min_value=0.6, saturation=0.5 )
            palette = np.array( palette )
            colors  = list( palette.T.flatten() )
        else:
            colors  = tab_colors
            assert len( colors ) >= n, f"not enough colors for {n} labels"

    pyplot.rcParams.update( { "font.size": 14 } )
    handles     = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, xlabels ) ]
    fig, ax     = pyplot.subplots( figsize=bfigsize )

    for i, xx in enumerate( x ):
        m, e    = res[ i ]
        ax.bar( xx, m, yerr=e, color=colors[ i ], width=width )
    ax.set_xticks( xpos, labels=xlabels, rotation=xlabel_rot )

    if y_range is not None:
        ax.set_ylim( bottom=y_range[ 0 ], top=y_range[ 1 ] )
        if len( y_range ) > 2:
            y_tick  = y_range[ -1 ]
        else:
            y_tick  = 0.1
        ax.set_yticks( np.arange( y_range[ 0 ], y_range[ 1 ]+0.01, y_tick ) )

    ax.set_ylabel( ylabel )

    fname       = basename + extension
    pyplot.margins( x=0 )
    pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_multi_bars(
        res_dict,
        labels,
        plot_titles,
        twin_plots  = 3,
        suptitle    = '',
        basename    = "plot_",
        h_legend    = False,
        width       = 2,
        colors      = None,
        y_range     = None ):
    """
    plotting bars of averaged results, optionally grouped, and over mu,tiple plots

    input:
        res_dict        [dict] of [dict] with higher keys as plot_titles, inner as labels, and (mean,std) values
        labels          [tuple] strings for the results in each subplots
        plot_titles     [tuple] strings for the titles of the subplots
        twin_plots      [int] plots should be grouped in twin_plots, 1 for no grouping
        suptitle        [str] superior title of the plot
        basename        [str] basename of the file with the plot
        h_legend        [bool] set the legend to be displayed horizontally
        width           [float] bars width
        colors          [list] with all necessary colors, in any Matplotlib allowed format, None for automatic
        y_range         [tuple] of min and max Y, and optionally Y tick
    """
    n_plots = len( plot_titles )
    n       = len( labels )
    assert n_plots == len( res_dict ), "mismatched results dict and subplots titles"
    x       = [  i + i // twin_plots for i in range( n ) ]
    if colors is None:
        if twin_plots > 1:
            palette = gen_colors( n // twin_plots, twin_plots, min_value=0.6, saturation=0.5 )
            palette = np.array( palette )
            colors  = list( palette.T.flatten() )
        else:
            colors  = tab_colors
            assert len( colors ) >= n, f"not enough colors for {n} labels"

    pyplot.rcParams.update( { "font.size": 14 } )
    handles         = [ Patch( facecolor=c, label=l ) for c,l in zip( colors, labels ) ]
    legend_inplot   = False             # when true uses the last suplot for the legends
    if n_plots > 4:
        nrows   = 2
        ncols   = n_plots // 2
        if n_plots % 2:
            if not h_legend:
                legend_inplot   = True
                ncols           += 1
        else:
            h_legend    = True          # in this case should force horizontal legends
    else:
        nrows   = 1
        ncols   = n_plots
        if not h_legend:
            legend_inplot   = True
            ncols           += 1
    fig, axs    = pyplot.subplots( nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True )
    axf         = axs.flat

    plot_axes   = axf[ : -1 ] if legend_inplot  else axf
    for g, ax in enumerate( plot_axes ):
        ax.grid( which='both', axis='y', color='#cccccc' )
        ax.set_axisbelow( True )
        data_dir    = res_dict[ plot_titles[ g ] ]
        for i, xx in enumerate( x ):
            m, e    = data_dir[ labels[ i ] ]
            ax.bar( xx, m, yerr=e, color=colors[ i ], width=width )
        ax.set_title( plot_titles[ g ] )
        ax.set_xticks( [] )
        if y_range is not None:
            ax.set_ylim( bottom=y_range[ 0 ], top=y_range[ 1 ] )
            if len( y_range ) > 2:
                y_tick  = y_range[ -1 ]
            else:
                y_tick  = 0.1
            ax.set_yticks( np.arange( y_range[ 0 ], y_range[ 1 ]+0.01, y_tick ) )

    axf[ 0 ].set_ylabel( "fraction of responses" )

    # in the case of legend in horizonal, use one or more rows depending on the number of labels,
    # and align with the first group if one rows, otherwise with the second group
    if h_legend:
        if n > 8:
            ncol        = n // twin_plots
            x_bbox      = 0.9
        else:
            ncol        = n
            x_bbox      = 0.0
        if nrows == 2:
            ax_0    = axs[ 1, 0 ]
        else:
            ax_0    = axs[ 0 ]
        ax_0.legend(
                handles         = handles,
                loc             = "upper left",
                bbox_to_anchor  = [ x_bbox, -0.03 ],
                ncol            = ncol,
                labelspacing    = labelspacing
        )
    else:
        axf[ n_plots ].set_axis_off()
        axf[ n_plots ].legend( handles=handles, loc="center", bbox_to_anchor=[0.5,.5], labelspacing=labelspacing )
    fname       = basename + extension
    pyplot.margins( x=0 )
    pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_radar( df, main_var, group, values=None, score="score", basename="radar", suptitle="", t_angle=90 ):
    """
    Generate a radar plot

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame
        main_var    [str] name of the main column of independent variables for which bars are aligned
        group       [str] one column of independent variables
        values      [list] of values to be used in the group column, None for all
        score       [str] one column with the dependent variable
        fname       [str] name of the output file
        suptitle    [str] plot title
    """
    label_offset    = 1.20  # Adjust to increase/decrease distance
    columns         = df.columns
    assert main_var in columns, f"there is no column named {main_var}"

    ticks       = [ 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8 ]

    # extract the categories to plot
    cat         = df[ main_var ].unique().tolist()
    n_cat       = len( cat )
    assert n_cat <= 10, f"cannot do radar plot for {n_cat} categories"

    poly_name   = []
    poly_data   = []

    # extract the labels of the group to plot
    assert group in columns, f"there is no column named {group}"
    if values is None:
        values      = df[ group ].unique().tolist()
    for v in values:
        dv      = df[ df[ group ] == v ]
        dvm     = dv.groupby( main_var, observed=True )[ score ].mean()
        poly_name.append( f"{v}" )
        poly_data.append( dvm.tolist() )

    # compute angle for each axis
    angles = np.linspace( 0, 2 * np.pi, n_cat, endpoint=False ).tolist()

    # close the polygons
    angles  += angles[ :1 ]
    for p in poly_data:
        p   += p[ :1 ]

    pyplot.rcParams.update( { "font.size": 14 } )

    fig, ax     = pyplot.subplots( figsize=radar_fsize, subplot_kw=dict( polar=True ) )

    for i, p in enumerate( poly_data ):
        l       = poly_name[ i ]
        ax.plot( angles, p, label=l, color=tab_colors[ i ], linewidth=2, linestyle=line_style[ i ] )
        ax.fill( angles, p, color=tab_colors[ i ], alpha=0.25 )

    ax.set_xticks( angles[:-1] )
    # ax.set_xticklabels( cat )
    ax.set_rgrids( ticks, angle=t_angle, fontsize=12 )
    ax.set_ylim(0, 2)

    # hide default labels
    ax.set_xticklabels( [] )
    # add custom labels with increased distance
    for angle, label in zip(angles[:-1], cat):
        ax.text(angle, label_offset, label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12)

    ax.yaxis.grid(True, color='gray', linestyle='dotted', linewidth=0.5)
    ax.xaxis.grid(True)

    ax.legend( loc='upper right', bbox_to_anchor=(1.2, 1.1) )
    fname       = f"{basename}{extension}"
    if len( suptitle ):
        pyplot.suptitle( suptitle, x=0.4 )
    pyplot.savefig( fname, bbox_inches="tight", pad_inches=0 )
    pyplot.close()
    pyplot.clf()


def plot_models_radar( df, mdf, main_var, group=None, fname="pl", t_angle=100 ):
    """
    Wrapper for executing plot_radar() on all models found in the dataset, together with a final
    plot with all models data, for all categories in the specified main independent variable

    params:
        df          [pandas.core.frame.DataFrame] the data in pandas DataFrame for all models
        mdf         [pandas.core.frame.DataFrame] the data in pandas separated by models
        main_var    [str] name of the main column of independent variables for which single plots are produced
        group       [str] column of independent variables
        fname       [str] name of the output file
        t_angle     [float] angle at which ticks are drawn
    """
    models      = mdf[ "model" ].unique().tolist()    # find all models used in the dataframe

    for m in models:
        dm      = mdf[ mdf[ "model" ] == m ]
        name    = f"{fname}_radar_{m}"
        plot_radar( dm, main_var, group=group, fname=name, t_angle=t_angle, suptitle=f"model {m}" )

    if len( models ) > 1:
        name    = f"{fname}_radar_all"
        plot_radar( df, main_var, group=group, fname=name, t_angle=t_angle, suptitle="all models" )
