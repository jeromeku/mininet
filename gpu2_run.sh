#!/bin/bash

THEANO_FLAGS='floatX=float32,device=gpu2,nvcc.fastmath=True' python $1
