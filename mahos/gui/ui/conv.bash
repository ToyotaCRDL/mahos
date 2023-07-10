#!/bin/bash

convert () {
    ff=$(basename $1 .ui).py
    pyuic6 $1 > $ff
}

if [[ $# -eq 0 ]]; then
    for f in *.ui; do
        convert $f
    done
else
    for f in "$@"; do
        if [[ ${f##*.} == "ui" ]]; then
            convert $f
        else
            echo "[ERROR] $f is not a ui file."
        fi
    done
fi
