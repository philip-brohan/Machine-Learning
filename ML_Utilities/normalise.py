# (C) British Crown Copyright 2019, Met Office
#
# This code is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#

# Provide default normalisation functions for some 20CR data.

def get_normalise_function(source='20CR2c',variable='prmsl'):
    """Provide a normalisation function to scale  a 20CR data field to mean=0 sd=1 (approximately).
 
     Args:
        source (:obj:`str`): Where to get the data from - at the moment, only '20CR2c' is supported .
        variable (:obj:`str`): Variable to use (e.g. 'prmsl')

     Returns:
        function which, when called with a numpy array as its argument, returns a normalised version of that array.
  
    Raises:
        ValueError: Unsupported source or variable.
    |
    """

    if source=='20CR2c':
        if variable=='prmsl':
            def normalise(x):
                x -= 101325
                x /= 3000
                return x
        else:
            raise ValueError("Don't know how to normalise %s" % variable)
    else:
        raise ValueError("Don't know how to normalise data from %s" % source)
    
    return normalise

def get_unnormalise_function(source='20CR2c',variable='prmsl'):
    """Provide an unnormalisation function to scale a 20CR data field back to its original units from the normalised representation with mean=0 sd=1 (approximately).
 
     Args:
        source (:obj:`str`): Where to get the data from - at the moment, only '20CR2c' is supported .
        variable (:obj:`str`): Variable to use (e.g. 'prmsl')

     Returns:
        function which, when called with a (normalised) numpy array as its argument, returns an unnormalised version of that array.
  
    Raises:
        ValueError: Unsupported source or variable.
    |
    """

    if source=='20CR2c':
        if variable=='prmsl':
            def unnormalise(x):
                x *= 3000
                x += 101325
                return x
        else:
            raise ValueError("Don't know how to unnormalise %s" % variable)
    else:
        raise ValueError("Don't know how to unnormalise data from %s" % source)
    
    return unnormalise


