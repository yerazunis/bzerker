//   Testset for the bzerker library implementing Mitchie's BOXES algorithm
//   Copyright 2017 W.S.Yerazunis, released under the GPL V2 or later
//
//
//  Learn to play tic-tac-toe by reinforcement learning.
//
//  Data structures: the board, which has a 0, a +1, or a +2 on each cell
//  Actions: 1 thru 9 (always allocated) - we generate a mask at runtime
//   to avoid choosing actions that are illegal (i.e. moves to
//   already-occupied cells on the board)
//  States- there are 3^9 = 19683 possible states for the tic-tac-toe
//   board, but not all of them are reachable.  We Don't Care (yes, the
//   classic Mitchie treatment uses reflectional and rotational symmetry for 314
//   unique boards.  We could do that, or we could just brute-force it,
//   at something like a 20:1 disadvantage.  But it's faster to code
//   this way, and far more general.
//  There are only 9 possible moves on each side, so ACTIONS = 9 (yes, some
//   are illegal.  We use the "mask" variable to specify which are legal,
//   and generate it at runtime rather than enumerating it for all 19K boards.
//
//
#include "bzerker.h"
#include <stdio.h>
#include <stdlib.h>
//    Parameters to vary for testing
#define STATES 19683
#define ACTIONS 9
#define TOKENS 1000
#define REPEATS   20000000
#define BATCHSIZE 1000000
#define WIN_ADD 1.0
#define WIN_MUL 1.0
#define LOSE_ADD (-1.0)
#define LOSE_MUL 1.0
#define DRAW_ADD 0.01
#define DRAW_MUL 1.0
#define MAX_TURNS 10
#define PRINT_EACHGAME 0

//   A few globals:
bz_brain *brain1, *brain2;   // our two competing brains
bz_chain *chain1, *chain2;
int gb[9]; // the game board
int *gamblers_ruins;
//   Some function definitions.
main ()
{
  printf ("Starting test 2 - learning tic-tac-toe\n");
  bz_init();
  
  printf (" Initializing two brains.  They'll alternate who goes first.\n");
  brain1 = bz_newbrain (BZ_BRAIN_QUANTIZED, STATES, ACTIONS, TOKENS);
  brain2 = bz_newbrain (BZ_BRAIN_QUANTIZED, STATES, ACTIONS, TOKENS);
  printf ("Got brains! pointers are %li and %li\n",(long) brain1,(long) brain2);

  printf (" I would like to play %d double-games of tic-tac-toe.  Against myself.\n", REPEATS);
  int reps, i, j, action, batch;
  action = 0;
  int log_0 [REPEATS/BATCHSIZE];
  int log_1 [REPEATS/BATCHSIZE];
  int log_2 [REPEATS/BATCHSIZE];
  int log_gr [REPEATS/BATCHSIZE];
  for (i = 0; i < (REPEATS/BATCHSIZE); i++) log_0[i] = log_1[i] = log_2[i] = log_gr[i] = 0;
  for (reps = 0; reps < REPEATS; reps++) {
    batch = reps/BATCHSIZE;
    gamblers_ruins = & (log_gr[batch]);
    //printf (" Starting game %d \n", reps);
    chain1 = bz_newchain (brain1);
    chain2 = bz_newchain (brain2);
    action = play_ttt ( brain1, chain1, brain2, chain2, gamblers_ruins);
    bz_killchain (chain1);
    bz_killchain (chain2);
    if (action == 0) { log_0[batch]++; };
    if (action == 1) { log_1[batch]++; };
    if (action == 2) { log_2[batch]++; };
    chain1 = bz_newchain (brain1);
    chain2 = bz_newchain (brain2);
    action = play_ttt ( brain2, chain2, brain1, chain1, gamblers_ruins);
    bz_killchain (chain1);
    bz_killchain (chain2);
    if (action == 0) { log_0[batch]++; };
    if (action == 1) { log_1[batch]++; };
    if (action == 2) { log_2[batch]++; };
  }
  printf ("\nFactors (M,A): W: %f %f  L: %f %f  D: %f %f\n",
	  WIN_MUL, WIN_ADD, LOSE_MUL, LOSE_ADD, DRAW_MUL, DRAW_ADD);
  printf ("Overall Results: \n Pttn          P1         P2        Draw    Underflow\n");
  long p50, p90, ctpp;  // 50% and 90% points for draws, underflows
  p50 = p90 = 999999999;
  ctpp = 0;
  for (batch = 0; batch < (REPEATS / BATCHSIZE); batch++) {
    printf (" %9d %9d %9d %9d %9d\n",
	    batch*BATCHSIZE, log_1[batch], log_2[batch],
	    log_0[batch], log_gr[batch] );
    if (log_1[batch] < log_0[batch] && log_2[batch] < log_0[batch]
	&& batch*BATCHSIZE < p50)
      p50 = batch*BATCHSIZE;
    if (10 * log_1[batch] < log_0[batch] && 10 * log_2[batch] < log_0[batch]
	&& batch*BATCHSIZE < p90)
      p90 = batch*BATCHSIZE;
    if (log_gr[batch] > 0) ctpp = batch * BATCHSIZE;
  }
  printf ("\n P50 at %d, P90 at %d, final underflow at %d \n", p50, p90, ctpp);
  printf ("All done.  That was fun.  Play more later.\n");
}

//    Determine victory conditions - returns either 0 (neither), -1 (draw),
//    1 (brain1 won, or 2 (brain2 won).
//    NOTE: it's perfectly possible to have different victory conditions for
//    brain1 and brain2.
//    We number the cells top to bottom, then left to right.  Note that this
//    is the ONLY PLACE THAT MATTERS!  Why?  Because our algorithm knows nothing
//    of the actual rules, only those state combinations when it wins!
//      0 1 2
//      3 4 5
//      6 7 8
//
int victory() {
  //  victory player 2
  if ((gb[0] == 2 ) && ( gb[1] == 2 ) && ( gb[2] == 2)) return 2;
  if ((gb[3] == 2 ) && ( gb[4] == 2 ) && ( gb[5] == 2)) return 2;
  if ((gb[6] == 2 ) && ( gb[7] == 2 ) && ( gb[8] == 2)) return 2;
  if ((gb[0] == 2 ) && ( gb[3] == 2 ) && ( gb[6] == 2)) return 2;
  if ((gb[1] == 2 ) && ( gb[4] == 2 ) && ( gb[7] == 2)) return 2;
  if ((gb[2] == 2 ) && ( gb[5] == 2 ) && ( gb[8] == 2)) return 2;
  if ((gb[0] == 2 ) && ( gb[4] == 2 ) && ( gb[8] == 2)) return 2;
  if ((gb[6] == 2 ) && ( gb[4] == 2 ) && ( gb[2] == 2)) return 2;
  
  //  victory player 1
  if ((gb[0] == 1 ) && ( gb[1] == 1 ) && ( gb[2] == 1)) return 1;
  if ((gb[3] == 1 ) && ( gb[4] == 1 ) && ( gb[5] == 1)) return 1;
  if ((gb[6] == 1 ) && ( gb[7] == 1 ) && ( gb[8] == 1)) return 1;
  if ((gb[0] == 1 ) && ( gb[3] == 1 ) && ( gb[6] == 1)) return 1;
  if ((gb[1] == 1 ) && ( gb[4] == 1 ) && ( gb[7] == 1)) return 1;
  if ((gb[2] == 1 ) && ( gb[5] == 1 ) && ( gb[8] == 1)) return 1;
  if ((gb[0] == 1 ) && ( gb[4] == 1 ) && ( gb[8] == 1)) return 1;
  if ((gb[6] == 1 ) && ( gb[4] == 1 ) && ( gb[2] == 1)) return 1;
  //   draw condition - all cells filled, no winner
  if (gb[1] != 0 && gb[2] != 0 && gb[3] != 0
      && gb[4] != 0 && gb[5] != 0 && gb[6] != 0
      && gb[7] != 0 && gb[8] != 0 && gb[0] != 0)
    return -1;
  //   No winner yet, keep playing!
  return 0;
}

//   GBS - Game Board State.   Convert the gameboard gb into an int state.
//  Simple conversion from an array via base-3 encoding.
long gbs(int gb[9]) {
  int rval;
  rval = (1 * gb[0]
	  + 3 * gb[1]
	  + 9 * gb[2]
	  + 27 * gb[3]
	  + 81 * gb[4]
	  + 243 * gb[5]
	  + 729 * gb[6]
	  + 2187 * gb[7]
	  + 6561 * gb[8]);
  return rval;
}

  
//   Play one game of tic-tac-toe, b1 versus b2
int play_ttt( bz_brain *b1, bz_chain *s1,
	      bz_brain *b2, bz_chain *s2,
	      int *gamblers_ruin) {
  long i, movecount, move, victor, state;
  char mask[9];
  int showboards;
  bz_brain *btemp;
  bz_chain *ctemp;
  showboards = 0;
  //   Start with a blank board.
  for (i = 0; i < 9; i++) gb[i] = 0;
  victor = 0;
  //   loop till someone wins (or not).
  movecount = 0;  //  for when we're only allowing 4 moves each side.
  while ( victor == 0 && movecount < MAX_TURNS ) {
    movecount++;
    //   get b1's next move
    state = gbs (gb);  // turn the gameboard into an integer "state"
    for (i = 0; i < 9; i++) mask[i] = !(gb[i]);  //create a mask of legal moves
    //
    //   TEST - see if restricting first player move to a symmetry move helps
    //   Results: with: 180k, 210K, 700K
    //         without: 240k, 270K, 410K
    //   Conclusion:  not a lot!
    if (movecount == -1) {
      for (i = 0; i < 9; i++) mask[i] = 0;
      mask[0] = mask[1] = mask[4] = 1;
    }
    move = bz_next_action (b1, state, mask, gamblers_ruin);
    if (showboards) printf ("Board: %d%d%d%d%d%d%d%d%d S: %d P%d moves %d \n",
        gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],gb[8],
			    state, 1, move);
    if (showboards) printf ("Mask:  %d%d%d%d%d%d%d%d%d \n",
       mask[0],mask[1],mask[2],mask[3],mask[4],mask[5],mask[6],mask[7],mask[8]);
    //   execute the move (if it's illegal, take next legal one.)
    //    Yes, that's suboptimal.  GROT GROT GROT
    bz_addtochain (s1, state, move);
    // printf ("executing move\n");
    execute_move (move, 1);
    //  check, did someone win?  Exit the while-loop if they did!
    victor =  victory();
    if (victor != 0) break;
    //   No winner, let b2 take a turn
    movecount++;
    if (movecount > MAX_TURNS) break;
    state = gbs (gb);
    for (i = 0; i < 9; i++) mask[i] = !(gb[i]);
    move = bz_next_action (b2, state, mask, gamblers_ruin);
    if (showboards) printf ( "Board: %d%d%d%d%d%d%d%d%d S: %d P%d moves %d \n",
	    gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],gb[8],
			     state, 2, move);
    if (showboards) printf ("Mask:  %d%d%d%d%d%d%d%d%d \n",
       mask[0],mask[1],mask[2],mask[3],mask[4],mask[5],mask[6],mask[7],mask[8]);
    //  execute that move
    bz_addtochain (s2, state, move);
    execute_move (move, 2);
    //  did someone win?
    victor = victory();
    if (victor != 0) break;
  }
  if (showboards)  fprintf (stderr, "victor: %d board: %d%d%d %d%d%d %d%d%d \n",
  	   victor,
  	   gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7],gb[8]);
  if (victor == 1) {
    if (PRINT_EACHGAME) fprintf (stderr, "A");
    bz_learnchain (b1, s1, WIN_ADD, WIN_MUL, 0);  //  add, then multiply
    bz_learnchain (b2, s2, LOSE_ADD, LOSE_MUL, 0);
    return 1 ;
  }
  if (victor == 2) {
    if (PRINT_EACHGAME) fprintf (stderr, "B");
    bz_learnchain (b2, s2, WIN_ADD, WIN_MUL, 0);
    bz_learnchain (b1, s1, LOSE_ADD, LOSE_MUL, 0);
    return 2;
  }
  //   No winner!
  if (PRINT_EACHGAME) fprintf (stderr, "X");
  bz_learnchain (b2, s2, DRAW_ADD, DRAW_MUL, 0);
  bz_learnchain (b1, s1, DRAW_ADD, DRAW_MUL, 0);
  return 0;
}

//    Take a tic-tac-toe move- action is which square to mark, value is 1 or 2
int execute_move (square, value) {
  int i, sq, legal;
  legal = square;
  //fprintf (stderr, "%d", legal);
  //  take the first legal move from this square.
  for (i = 0; i < 9; i++) {
    sq = (square + i) % 9;
    if (gb[sq] == 0) {
      gb[sq] = value;
      return (0);
    }
  }
  //  Not a single legal move exists. Return failure.
  //return (1);
  fprintf (stderr, "\nILLEGAL MOVE SUGGESTED, numbr %d of %d\n", legal, square);
  exit(0);
}
