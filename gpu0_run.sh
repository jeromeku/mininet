#!/bin/bash

THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python $1
