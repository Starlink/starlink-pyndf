
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
        self.pos   = indf.aread('Centre',iaxis)
        self.var   = indf.aread('Variance',iaxis)
        self.width = indf.aread('Width',iaxis)
        self.label = indf.acget('Label',iaxis)
        self.units = indf.acget('Units',iaxis)
