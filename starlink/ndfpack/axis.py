
class Axis(object):
    """
    Represents an NDF axis.

    Not all attributes are guaranteed to be defined.
    """

    def __init__(self, indf, iaxis):
        """Initialise an NDF axis."""

        #: positions of centres of pixels
        self.pos   = indf.aread('Centre',iaxis)

        #: variances of positions of centres of pixels
        self.var   = indf.aread('Variance',iaxis)

        #: widths of pixels
        self.width = indf.aread('Width',iaxis)

        #: character string label
        self.label = indf.acget('Label',iaxis)

        #: units of the axis
        self.units = indf.acget('Units',iaxis)
