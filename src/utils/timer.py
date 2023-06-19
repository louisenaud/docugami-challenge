"""
File:        timer.py
Created by:  Louise Naud
On:          6/19/23
At:          11:57 AM
For project: docugami-challenge
Description:
Usage:
"""
import logging
import time
from contextlib import contextmanager


@contextmanager
def timer(name, disable=False):
    """Simple timer as context manager."""

    start = time.time()
    yield
    if not disable:
        logging.info(f'[{name}] done in {(time.time() - start) * 1000:.1f} ms')
