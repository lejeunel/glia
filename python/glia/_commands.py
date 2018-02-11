"""
Commands for use with GLIA processing.

Jeffrey Bush, 2017, NCMIR, UCSD
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod 
from pysegtools.general.utils import ravel
def unique(itr): return tuple(set(itr))

class Cmd(object):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        self.args = args
        self.__skipper = kwargs.pop('skip', (lambda s,a:False))
        # TODO: task keyword args
        if len(kwargs) != 0: raise ValueError('unsupported keyword arguments: %s'%kwargs)
        self.skip = False
    @abstractmethod
    def add_tasks(self, tasks, n): pass
    @property
    def arguments(self):
        return (a for a in self.args if isinstance(a, Argument))
    def add_parser_arg(self, parser): 
        for arg in self.arguments: arg.add_parser_arg(parser)
    def apply_settings(self, settings, args):
        for arg in self.arguments: arg.apply_settings(settings, args)
        self.skip = self.__skipper(settings,args)
    def cmd(self, i): return tuple(ravel(a if isinstance(a, basestring) else a.args(i) for a in self.args))
    def inputs(self, i): return unique(ravel(a.inputs(i) for a in self.arguments))
    def outputs(self, i): return unique(ravel(a.outputs(i) for a in self.arguments))
    @property
    def settings(self): return unique(ravel(a.settings for a in self.arguments))

class SingleCmd(Cmd):
    """A command that is executed once"""
    def add_tasks(self, tasks, n):
        if self.skip: return
        tasks.add(self.cmd(n), inputs=self.inputs(n), outputs=self.outputs(n), settings=self.settings)

class RepeatedCmd(Cmd):
    """A command that is executed for each slice"""
    def add_tasks(self, tasks, n):
        if self.skip: return
        settings = self.settings
        for i in xrange(n):
            tasks.add(self.cmd(i), inputs=self.inputs(i), outputs=self.outputs(i), settings=settings)

class Argument(object):
    """Any argument in a command line that is dynamic."""
    #pylint: disable=unused-argument, no-self-use
    def args(self, i):
        """
        Gets the list of argument(s) to put on the command line where i is either the number of slices (for a
        SingleCmd) or the current slice number (for a RepeatedCmd).. Default implementation always returns the
        empty list.
        """
        return ()
    def inputs(self, i):
        """
        Gets the input file(s) where i is either the number of slices (for a SingleCmd) or the current slice number
        (for a RepeatedCmd). Default implementation always returns the empty list.
        """
        return ()
    def outputs(self, i):
        """
        Gets the output file(s) where i is either the number of slices (for a SingleCmd) or the current slice number
        (for a RepeatedCmd). Default implementation always returns the empty list.
        """
        return ()
    @property
    def settings(self):
        """
        Gets the settings this argument utilizes to determine its outputs in the other methods. Default implementation
        always returns the empty list.
        """
        return ()
    def add_parser_arg(self, parser):
        """Adds any necessary arguments to the given ArgumentParser. Default implementation does nothing."""
        pass
    def apply_settings(self, settings, args):
        """
        Given the settings and argument namespace adjust the internal values of this object so that subsequent calls
        to args, inputs, and outputs behave appropiately. Default implementation does nothing.
        """
        pass

class Setting(Argument):
    """
    An argument in a command line that gets its value from an argument passed to the program but this does not
    actually add the argument to the argparser, just references it. 
    """
    def __init__(self, name, default):
        self.name = name
        self.default = default
        self.value = default
    @property
    def settings(self): return (self.name,)
    def args(self, i):
        x = self.value
        if isinstance(x, (tuple, list)): return tuple(str(y) for y in x)
        if isinstance(x, bool): return 'true' if x else 'false'
        return str(x)
    def apply_settings(self, settings, args):
        self.value = self._conv_value(settings.get(self.name, self.default))
    def _conv_value(self, x): return x #pylint: disable=no-self-use

class Parameter(Setting):
    """
    An argument in a command line that gets its value from an argument passed to the program and is responsible
    for adding it to the argparser.
    """
    def __init__(self, name, default, typ, hlp):
        if '&' in name:
            self.short_opt = name[name.index('&')+1]
            name = name.replace('&', '')
        else: self.short_opt = None
        super(Parameter, self).__init__(name, default)
        self.type = typ
        self.help = hlp
    def add_parser_arg(self, parser):
        parser.add_argument(*self._parser_flags, **self._parser_arg_params)
    @property
    def _parser_flags(self):
        return ('-'+self.short_opt, '--'+self.name) if self.short_opt else ('--'+self.name,)
    @property
    def _parser_arg_params(self):
        return {'default': self.default, 'type': self.type, 'help': self.help, 'metavar': self._parser_metavar}
    @property
    def _parser_metavar(self):
        if self.short_opt: return None
        t,d = self.type, self.default
        seq = isinstance(d, (list,tuple))
        if t == int or isinstance(d, int) or seq and isinstance(d[0], int): return 'N'
        if t == float or isinstance(d, float) or seq and isinstance(d[0], float): return 'F'
        return 'X'

class BoolParameter(Parameter):
    """A boolean parameter, one that if it is present is the value 'true' otherwise it is 'false'."""
    def __init__(self, name, hlp):
        super(BoolParameter, self).__init__(name, False, None, hlp)
    @property
    def _parser_arg_params(self):
        return {'action':'store_true', 'help': self.help}
    def _conv_value(self, x): return bool(x)

class ListParameter(Parameter):
    """A parameter that takes many arguments."""
    def __init__(self, name, default, typ, nargs, hlp):
        # nargs can be one of the ArgParser nargs values:
        #   an integer for exactly that many values
        #   '?' for 0 or 1 values
        #   '*' for 0 or more values
        #   '+' for 1 or more values
        #   argparse.REMAINDER
        # or a tuple/list of integers (e.g. (2,3) )
        from argparse import REMAINDER
        from numbers import Integral
        from collections import Sequence
        super(ListParameter, self).__init__(name, list(default), typ, hlp)
        raw_nargs = nargs
        if nargs not in ('?', '+', '*', REMAINDER) and not isinstance(nargs, Integral):
            if not isinstance(nargs, Sequence): raise ValueError('nargs')
            mn = unique(nargs)
            if mn < 0: raise ValueError('nargs')
            raw_nargs = '*' if mn == 0 else '+'
        self.raw_nargs = raw_nargs
        self.nargs = nargs
    def apply_settings(self, settings, args):
        from argparse import ArgumentTypeError
        super(ListParameter, self).apply_settings(settings, args)
        if self.nargs != self.raw_nargs and len(self.value) not in self.nargs:
            raise ArgumentTypeError('wrong number of values for %s, got %d values'%(self.name, len(self.value)))
    @property
    def _parser_arg_params(self):
        return {'default':self.default, 'type':self.type, 'nargs':self.raw_nargs, 'help': self.help, 'metavar': self._parser_metavar}
    def _conv_value(self, x): return tuple(x)

class EnumParameter(Parameter):
    """A parameter that takes a value from a list of values and sends a numerical index to the underlying program."""
    def __init__(self, name, default, choices, hlp):
        super(EnumParameter, self).__init__(name, default, None, hlp)
        self.choices = choices
    @property
    def _parser_arg_params(self):
        return {'default':self.default, 'choices':self.choices, 'help': self.help}
    def _conv_value(self, x): return self.choices.index(x) + 1


class Condition(Argument):
    """
    An argument that checks a condition and then forwards onto another Argument if it is true or false (if a third
    value is not given to the constructor than if false nothing is output).
    """
    argx = None
    def __init__(self, cond, arg1, arg2=None, extra_settings=()):
        self.cond = cond
        self.arg1 = arg1
        self.arg2 = arg2
        self.extra_settings = extra_settings
    def args(self, i): return () if self.argx is None else self.argx.args(i)
    def inputs(self, i): return () if self.argx is None else self.argx.inputs(i)
    def outputs(self, i): return () if self.argx is None else self.argx.outputs(i)
    @property
    def settings(self):
        settings = unique(ravel(self.extra_settings))
        settings += unique(ravel(self.arg1.settings))
        if self.arg2 is not None: settings += unique(ravel(self.arg2.settings))
        return settings
    def add_parser_arg(self, parser):
        self.arg1.add_parser_arg(parser)
        if self.arg2 is not None: self.arg2.add_parser_arg(parser)
    def apply_settings(self, settings, args):
        self.arg1.apply_settings(settings, args)
        if self.arg2 is not None: self.arg2.apply_settings(settings, args)
        self.argx = self.arg1 if self.cond(settings, args) else self.arg2

    
class File(Argument):
    """An argument that represents a single file on the computer - use InputFile or OutputFile depending on the usage."""
    def __init__(self, path):
        from os.path import dirname
        self.folder = dirname(path)
        self.path = path
    def args(self, n): return self.path #pylint: disable=arguments-differ,unused-argument
class InputFile(File):
    def inputs(self, n): return self.path #pylint: disable=arguments-differ,unused-argument
class OutputFile(File):
    def outputs(self, n): return self.path #pylint: disable=arguments-differ,unused-argument

class MultiFile(Argument):
    """
    A multi-slice image either represented by a bunch of 2D files or a single 3D file depending on the 3D mode.
    """
    threeD = False
    @property
    def settings(self): return ('threeD',)
    def apply_settings(self, settings, args): self.threeD = settings.get('threeD', False)
    
class FileSet(MultiFile):
    """
    An argument that represents a set of files on the computer where a SingleCmd will get a list of all of the
    files, each with a different index. Use InputSet or OutputSet depending on the usage.
    
    In 3D mode this represents a single 3D file.
    """
    def __init__(self, folder, ext):
        self.folder = folder
        self.ext = ext
    def files(self, n):
        from os.path import join
        if self.threeD: return ['%s.%s'%(self.folder,self.ext)]
        return [join(self.folder, '%04d.%s'%(i,self.ext)) for i in xrange(n)]

class InputSet(FileSet):
    #pylint: disable=arguments-differ
    def args(self, n): return self.files(n)
    def inputs(self, n): return self.files(n)
class OutputSet(FileSet):
    #pylint: disable=arguments-differ
    def args(self, n): return self.files(n)
    def outputs(self, n): return self.files(n)

class InputImageStack(InputSet):
    """
    An argument that represents an input "image stack" argument where the entire set of files is grouped into a
    single argument.

    In 3D mode this represents a single 3D file.
    """
    def args(self, n):
        from os.path import join
        if self.threeD: return ['%s.%s'%(self.folder,self.ext)]
        return (join(self.folder, '####.%s'%self.ext), '0', '1', '%d'%(n-1))
class OutputImageStack(OutputSet):
    """
    An argument that represents an output "image stack" argument where the entire set of files is grouped into a
    single argument.

    In 3D mode this represents a single 3D file.
    """
    def args(self, n):
        from os.path import join
        if self.threeD: return ['%s.%s'%(self.folder,self.ext)]
        return (join(self.folder, '####.%s'%self.ext), '0', '1')
    
class FileSeries(MultiFile):
    """
    An argument that represents a set of files on the computer where a RepeatedCmd will get a single file with a
    different index. Use InputSeries or OutputSeries depending on the usage.
    """
    def __init__(self, folder, ext):
        self.folder = folder
        self.ext = ext
    def args(self, i):
        from os.path import join
        if self.threeD: return ['%s.%s'%(self.folder,self.ext)]
        return join(self.folder, '%04d.%s'%(i,self.ext))
class InputSeries(FileSeries):
    def inputs(self, i): return self.args(i)
class OutputSeries(FileSeries):
    def outputs(self, i): return self.args(i)






# These are very specific to GLIA    
class Model(Argument):
    path = None
    def __init__(self, inpt=True, name='model'):
        self.name = name
        self.inpt = inpt
    def args(self, i): return self.path
    def inputs(self, i):  return self.path if self.inpt else ()
    def outputs(self, i): return () if self.inpt else self.path
    def apply_settings(self, settings, args):
        from os.path import relpath
        self.path = relpath(getattr(args, self.name), args.temp) # TODO: use os.path.commonprefix to improve results
class Masks(InputSeries):
    has_masks = False
    def __init__(self, arg='-m', folder='msk', ext='mha'):
        self.arg = arg
        super(Masks, self).__init__(folder, ext)
    def args(self, i):
        return (self.arg, super(Masks, self).args(i)) if self.has_masks else ()
    def inputs(self, i):
        return super(Masks, self).args(i) if self.has_masks else ()
    def apply_settings(self, settings, args):
        self.has_masks = getattr(args, self.folder+'s') is not None
