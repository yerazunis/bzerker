#   Makefile for the bzerker library implementing Mitchie's BOXES algorithm
#    Copyright 2017 W.S.Yerazunis, released under the GPL V2 or later

PROJECTNAME = bzerker

FILES = bzerker.c bzerker.h Makefile

DEBUG_FLAG = -g -O0

all: libbzerker

libbzerker: bzerker.c bzerker.h
#	cc DEBUG_FLAG bzerker.c

balltrack: bzerker.c bzerker.h balltrack.c
	cc bzerker.c balltrack.c -o balltrack

tictactoe: bzerker.c tictactoe.c  bzerker.h
	cc $(DEBUG_FLAG) bzerker.c tictactoe.c -o tictactoe

bzerker_test: bzerker.c bzerker_test.c  bzerker.h
	cc bzerker.c bzerker_test.c -o bzerker_test

bzerker_test1: bzerker.c bzerker_test1.c  bzerker.h
	cc bzerker.c bzerker_test1.c -o bzerker_test1

bzerker_test2: bzerker.c bzerker_test2.c  bzerker.h
	cc $(DEBUG_FLAG) bzerker.c bzerker_test2.c -o bzerker_test2

bzerker_test3: bzerker.c bzerker_test3.c  bzerker.h
	cc $(DEBUG_FLAG) bzerker.c bzerker_test3.c -o bzerker_test3

bzerker_test4: bzerker.c bzerker_test4.c  bzerker.h
	cc $(DEBUG_FLAG) bzerker.c bzerker_test4.c -o bzerker_test4

