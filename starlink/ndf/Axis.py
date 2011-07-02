from ._ndf import *

class Axis(object):
    """
    Represents an NDF axis.

    Attributes (not all of which are guaranteed to be defined)

    pos    -- positions of centres of pixels
    var    -- variances of positions of centres of pixels
    width  -- widths of pixels
    label  -- character string label.
    units  -- units of the axis
    """

    def __init__(self, indf, iaxis):
        """Initialise an NDF axis."""
        self.pos   = ndf_aread(indf,'Centre',iaxis)
        self.var   = ndf_aread(indf,'Variance',iaxis)
        self.width = ndf_aread(indf,'Width',iaxis)
        self.label = ndf_acget(indf,'Label',iaxis)
        self.units = ndf_acget(indf,'Units',iaxis)
