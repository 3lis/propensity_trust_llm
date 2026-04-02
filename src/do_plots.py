"""
#####################################################################################################################

    HURST project - 2024

    wrapper for executing plots on existing runs


#####################################################################################################################
"""


import  os
import  sys
import  plot

dir_res             = '../res'
dump_file_name      = 'results.pickle'
dir_plot            = 'plot'
plot_name           = 'plt_'

runs                = [
		"24-09-11_17-26-25",
		"24-09-11_19-27-13",
		"24-09-11_19-45-37",
		"24-09-11_22-34-13",
		"24-09-11_23-26-44",
		"24-09-12_04-20-01",
		"24-09-12_05-13-22",
		"24-09-12_09-58-48",
		"24-09-12_11-08-30",
		"24-09-12_12-37-16",
]

WINDOW              = 20

def plot_run( run ):
    dir_current = os.path.join( dir_res, run )
    if not os.path.isdir( dir_current ):
        print( f"run {dir_current} not found, skipping" )
        return False
    dir_out     = os.path.join( dir_current, dir_plot )
    if not os.path.isdir( dir_out ):
        os.makedirs( dir_out )
    basename    = os.path.join( dir_out, plot_name )
    pickle_file = os.path.join( dir_current, dump_file_name )
    if not os.path.isfile( pickle_file ):
        print( f"memory dump {pickle_file} not found, skipping" )
        return False
    plot.plot_success_history( pickle_file, basename=basename, window=WINDOW )
    plot.plot_accuracy_history( pickle_file, basename=basename, window=WINDOW )
    return True


def plot_runs():
    for run in runs:
        plot_run( run )

"""
if the file is executed, run the main plotting function
"""
if __name__ == '__main__':
    plot_runs()
