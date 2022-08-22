"""
Run all of the scripts
"""

import os

if not os.path.exists('../main/figures'):
    os.makedirs('../main/figures')

import relaxed
import tight
import private_equity
