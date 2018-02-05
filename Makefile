#   Makefile for the bzerker library implementing Mitchie's BOXES algorithm
#    Copyright 2017 W.S.Yerazunis, released under the GPL V2 or later

PROJECTNAME = bzerker

FILES = bzerker.c bzerker.h Makefile

DEBUG_FLAG = -g -O0

all: libbzerker

libbzerker: bzerker.c bzerker.h
#	cc DEBUG_FLAG bzerker.c

balltrack: bzerker.c bzerker.h balltrack.c
	cc bzerker.c balltrack.c -lm -o balltrack

tictactoe: bzerker.c tictactoe.c  bzerker.h
	cc $(DEBUG_FLAG) bzerker.c tictactoe.c -lm -o tictactoe

