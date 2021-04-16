#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter notebook --ip $(hostname -f) --no-browser
