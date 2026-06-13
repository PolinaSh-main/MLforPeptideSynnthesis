from .categorical_metadata import CategoricalMetadataFeature


class CouplingAgentFeature(CategoricalMetadataFeature):
    """One-hot encoding of the synthesis coupling agent (e.g. HATU, PyAOP)."""

    name = "coupling_agent"
    column = "coupling_agent"
