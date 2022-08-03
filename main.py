"""
Run all of the scripts
"""

import os

if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

import benchmark
import collateral
import private_equity
