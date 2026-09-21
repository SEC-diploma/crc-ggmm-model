#!/usr/bin/env bash
# Fill these paths before running Stages 01–04.
export THREADS=16
export STAR_GENOME=/path/to/STAR/GRCh38
export HOST_BT2=/path/to/bowtie2/GRCh38
export HOST_MM2=/path/to/GRCh38.mmi
export KRAKEN_DB=/path/to/kraken2/standard
export SILVA_TRAIN=/path/to/silva_nr99_v138.1_train_set.fa.gz
export HOST_OUT=./results/01_host
export MICRO_OUT=./results/02_microbe
