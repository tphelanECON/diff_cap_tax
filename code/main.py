"""
Run all of the scripts necessary to replicate the figures in the paper
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

import tight
import relaxed
import private_equity
