"""Thin launcher script for PyInstaller packaging & direct execution of SpoolGenie Desktop."""

from __future__ import annotations

import multiprocessing

from spoolgenie.desktop import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
