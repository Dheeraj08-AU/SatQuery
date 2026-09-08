"""pytest configuration — suppress known library-internal FutureWarnings."""

import warnings
import pytest


@pytest.fixture(autouse=True)
def suppress_grounding_dino_future_warning():
    """
    transformers fires this FutureWarning from inside post_process_grounded_object_detection
    when it builds the output dict. Our code already uses `text_labels`; the warning
    is a library-internal notice and adds no actionable signal to our test output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The key `labels` is will return integer ids",
            category=FutureWarning,
        )
        yield
