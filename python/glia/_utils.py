"""
Utilities functions.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

def get_delim(fn):
    """
    Get the expected delimiter for a data file based on the extension of the filename:
        ','   .csv
        '\t'  .tsv .tab
        ' '   otherwise
    The filename may also have a .gz or .bz2 extension which will be ignored.
    """
    fn = fn.lower()
    if   fn.endswith('.gz'): fn = fn[:-3]
    elif fn.endswith('.bz2'): fn = fn[:-4]
    if fn.endswith('.csv'): return ','
    if fn.endswith('.tsv') or fn.endswith('.tab'): return '\t'
    return ' '
    
def load_txt(fn):
    """
    Loads an array from a text file which is either comma or white-space seperated values based on
    the file extension. 
    """
    from numpy import loadtxt
    delim = get_delim(fn)
    return loadtxt(fn, delimiter=None if delim.isspace() else delim)

def save_txt(fn, arr):
    """
    Svaes an array to a text file which is either comma, tab, or space seperated values based on
    the file extension. 
    """
    from numpy import savetxt
    savetxt(fn, arr, delimiter=get_delim(fn))

def norm_rows(X, stats=None):
    """
    Normalize the rows of the data.
    
    X is an M-by-N array to normalize. It is modified in-place. The data is not returned.
    
    If stats is given it must be a sequence of 2 M-length arrays for the min and max statistics to
    normalize by instead of calculating them from the data itself.
    
    Returns the statistics used to normalize. If stats is given, it is just a tuple of the 2 values
    given. Otherwise it is the values calcuated from the data itself.
    """
    # Note: originally this normalized by the mean and standard deviation first like:
    #    Xmean,Xstd = X.mean(1),X.std(1,ddof=1)
    #    Xstd[Xstd == 0] = 1
    #    X -= Xmean[:,None]
    #    X /= Xstd[:,None]
    # However those calculations are actually useless if just re-normalizing.
    Xmin,Xmax = (X.min(1),X.max(1)) if stats is None else stats
    X -= Xmin[:,None]
    D = Xmax - Xmin
    D[D == 0] = 1
    X /= D[:,None]
    return Xmin,Xmax

def fix_std_buffering():
    """Re-opens the stdout and stderr file handles with no buffering."""
    from sys import stdout, stderr
    if hasattr(stderr, 'fileno'):
        try:
            from os import fdopen
            stdout = fdopen(stdout.fileno(), 'w', 0)
            stderr = fdopen(stderr.fileno(), 'w', 0)
        except StandardError: pass

def open_image_stack(cmd, nchannels=(1,)):
    """
    Used as a type for ArgumentParser.add_argument. This accepts grayscale image stacks only.
    The resulting value in the Namespace will be a tuple of the command line arguments and the
    FileImageStack object.
    """
    import shlex
    from pysegtools.images.io import FileImageStack
    from argparse import ArgumentTypeError
    
    args = shlex.split(cmd)
    try: ims = FileImageStack.open_cmd(args)
    except StandardError as e: raise ArgumentTypeError("failed to open '"+cmd+"': "+str(e))
    
    for im in ims:
        dt = im.dtype
        nc = dt.shape[0] if len(dt.shape) else 1
        if nc not in nchannels: raise ArgumentTypeError("an image in the stack '"+cmd+"' contains an invalid number of channels (did you provide an RGB image instead of a grayscale one?)")
    
    return (args, ims)
    
def open_image_stack_lbl(cmd):
    """Like open_image_stack except that it accepts grayscale and RGB images."""
    return open_image_stack(cmd, (1,3))
        
def create_image_stack(cmd):
    """
    Used as a type for ArgumentParser.add_argument. It returns just the command line arguments.
    """
    import shlex
    from pysegtools.images.io import FileImageStack
    from argparse import ArgumentTypeError
    args = shlex.split(cmd)
    try: FileImageStack.create_cmd(args, None)
    except StandardError as e: raise ArgumentTypeError("cannot parse the image stack command '"+cmd+"': "+str(e))
    return args

def temp_dir(x):
    """Function for use with ArgumentParser.add_argument for the temporary directory."""
    from os.path import abspath
    from pysegtools.general.utils import make_dir
    from argparse import ArgumentTypeError
    x = abspath(x)
    if not make_dir(x): raise ArgumentTypeError("error creating temporary directories")
    return x
        
def rusage_log(x):
    """Function for use with ArgumentParser.add_argument for the rusage log."""
    from sys import stderr
    from os.path import abspath
    try:
        from pysegtools.general.os_ext import wait4 # make sure wait4 is available #pylint: disable=unused-variable
        rl = abspath(x)
    except ImportError: print("warning: system does not support recording resource usage, 'rusage' argument ignored.", file=stderr)
    try:
        with open(rl, 'w'): pass # Make sure that the file can actually be opened
    except IOError: print("warning: rusage log file '%s' cannot be opened for writing, 'rusage' argument ignored."%x, file=stderr)
    return rl

def int_check(mn=None, mx=None):
    """
    Returns a function that can be used as the type for ArgumentParser.add_argument. This converts
    the argument to an int just like `int` would be allows for specifying a min and/or max value
    allowed on it as well.
    """
    def __int(x):
        from argparse import ArgumentTypeError
        try: x = int(x, 10)
        except ValueError: raise ArgumentTypeError("the value '"+x+"' is not a valid integer")
        if mn is not None and x < mn: raise ArgumentTypeError("the value %d is not >=%d"%(x,mn))
        if mx is not None and x > mx: raise ArgumentTypeError("the value %d is not <=%d"%(x,mx))
        return x
    return __int

def float_check(mn=None, mx=None):
    """
    Returns a function that can be used as the type for ArgumentParser.add_argument. This converts
    the argument to a float just like `float` would be allows for specifying a min and/or max value
    allowed on it as well.
    """
    def __float(x):
        from argparse import ArgumentTypeError
        try: x = float(x)
        except ValueError: raise ArgumentTypeError("the value '"+x+"' is not a valid floating-point number")
        if mn is not None and x < mn: raise ArgumentTypeError("the value %f is not >=%f"%(x,mn))
        if mx is not None and x > mx: raise ArgumentTypeError("the value %f is not <=%f"%(x,mx))
        return x
    return __float

