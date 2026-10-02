"""Environment failures propagated to experiment and evaluation callers."""


class EnvironmentUnavailable(RuntimeError):
    pass


class PatientAccessError(ValueError):
    pass


class UnsupportedDecisionPoint(ValueError):
    pass
