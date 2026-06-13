from .numeric_metadata import NumericMetadataFeature


class TempCouplingFeature(NumericMetadataFeature):
    """Scalar coupling temperature for the synthesis (NaN if not recorded)."""

    name = "temp_coupling"
    column = "temp_coupling"
