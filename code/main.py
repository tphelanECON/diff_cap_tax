"""
Run all of the scripts necessary to replicate the figures in "On the Optimality of Differential Asset Taxation."
"""

import os
if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

import tight
import relaxed
