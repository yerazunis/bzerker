//   BERZERKER - an implementation of Michie's 1960 BOXES algorithm for AI.
//   as popularized by Martin Gardner in Scientific American in 1961 and
//   then used as a plot device by Fred Saberhagen in his Berserker story
//   "Without A Thought" in 1963, which is where I first read of it sometime
//   around 1968.
//
//   Copyright W. S. Yerazunis; released under the GPL version 2 or later.
//
//   Version 0.01 - Just Testing Some Ideas I've Had Since I Was Twelve.
//
//
//   The Algorithm (expressed as a game with 1 or more turns):
//      
//    INIT: for each possible situation in a game:
//          create a "box";
//          for each possible move in that game situation:
//             fill the box with ten (or so) tokens for each one of the \
//             legal moves that can be made in that game situation (the
//             tokens say what move they represent);
//
//    STEP: find the box that corresponds to the current game situation;
//          from that box, randomly choose one of the tokens;
//          mark that token as "in use".
//          submit the move corresponding to that token as the turn.
//          (*) if there are no tokens left in the box, either
//              resign the game OR
//              put in a new batch of tokens, and pick one randomly.
//
//    LEARN: once the game is over and we know how well we did:
//          for each token that was "in use":
//             if we did well, add one or more additional tokens of that
//                type, how many depends on how well we did.
//             if we did poorly, we delete that token
//             if we were "just a draw" we neither add nor delete
//             then we unmark the "in use" flag.
//
//   Apparently Michie's original work actually was carried out with
//   pencil and paper; Martin Gardner's implementation used M&M candy,
//   and Fred Saberhagen's used plastic beads.   We can do better, by
//   using fuzzy sets to represent the move distributions.
//
//   Note that this algorithm does NOT use backpropagation, merely end-to-end
//   acceptability for an enumerated set of moves.   Note also that in
//   the limit of a very large number of training passes and single-turn
//   games, the algorithm convergs to the K-nearest-neighbor answer.
//
// EXTENSIONS (maybe now, maybe later, maybe never):
//
//   Mitchie's work depended on the totally discrete nature of the game
//   being learned; can we extend the algorithm to either nearest-seen
//   or interpolation, on input and/or output?
//
//   How about non-discrete (i.e. continuous) input state variables?  
//
//   How small can we make it?   DSPic-small?  Arduino-small?   
//
//   For large games, can we break down the situation (i.e. the state) into
//   a bunch of chunks, each chunk to be treated identically and then the
//   final action taken from the ensemble result?
//
//   The berzerker API also allows multiple simultaneous copies of the
//   algorithm to run simultaneously with different data (and with luck,
//   someday it may even run reentrantly so multicore processing may be
//   eventually possible someday.  But until we make sure it's
//   reentrant, stick with one thread, OK?)
//
// THE BERZERKER API
//
//  berzerker types and data structures:
//
//     void *bz_brain  -  an opaque pointer that holds the structs for
//                        the boxes; this is the learned solver.
//
//     void *bz_block  -  an opaque pointer that is allocated once and
//                        reusable to hold the actions made; this can be used
//                        for learning.
//
//     void *bz_chain  -  an opaque pointer that owns a malloc/free list
//                        of actions made; this can be used for learning.
//
//            Note that bz_block and bz_chains (or a mixture) can both be
//            used for learning.  They have different time nd memory costs
//            for adding, learning, and clearing:
//
//              bz_blocks use a fixed-size array roughly the same size as the
//              bz_brain structure.  Adding an action is constant time and
//              very fast (just two array dereferences); learning and clearing
//              require a scan of all cells which is proportional to the
//              number of problem states * number of possible actions at
//              each state.
//
//              bz_chains use a LIFO linked list, and start out as a very
//              small fixed-size structure.  Adding an action is constant time
//              but relatively slower than for blocks (there's a malloc()
//              required) but learning is very fast (one array dereference)
//              and clearing requires time proportional to the number of
//              actions stored (and does a free() on each link).
//
//            That said, the only thing that changes between blocks and chains
//            is the speed/memory tradeoff.  The computed results should
//            be identical (if they aren't, that's a bug!).
// 
//
/////////////////////////////////////////////////////////////////////
//     Things we gotta have!
#include <stdio.h>

/////////////////////////////////////////////////////////////////////
//
//       The version.   Please change if you hack the code.
#define BZ_VERSION_STRING "bzerker 20170523.WSY / First stake in the ground"

/////////////////////////////////////////////////////////////////////
//      trace message prefix
#define BZ_TRACEPREFIX "BZ_"

///////////////////////////////

/////////////////////////////////////////////////////////////////////////
//
#define BZ_BRAIN_QUANTIZED 0
 
/////////////////////////////////////////////////////////////////////////
//
//      The typedefs
//
/////////////////////////////////////////////////////////////////////////

typedef struct my_bz_state {
  long len;
  float *actions;   //   array of actions to be considered; note it's dense.
                    //   Sparsity will be handled in another level.
} bz_state;

typedef struct my_bz_brain {
  int braintype;        //  0 = discrete,  >0 = ???
  int maxstates;
  int maxactions;
  int starting_tokens;
  bz_state **states;  //  because C has only 1D dynarrays!
} bz_brain;

typedef struct my_bz_block {
  long totalcount;
  bz_brain *brain;
  long *tokens;  //  1D array workaround
} bz_block;

typedef struct bz__chel { // INTERNAL USE ONLY...chain element --> chel.
  long state;
  long action;
  struct bz__chel *next; 
} bz__chel;

typedef struct my_bz_chain {
  long totalcount;
  bz__chel *chels;
} bz_chain;

////////////////////////////////////////////////////////////////////////
//
//      The function definitions
//
////////////////////////////////////////////////////////////////////////

//     initialize the system
void bz_init ();

//     get the version (remember to free() the result)
char *bz_version ();

//     get status?   What does this do?
char *bz_status ();

//     Create a new brain.
bz_brain *bz_newbrain (
		       int braintype,   //  0 => discrete brain.  >0 => ????
		       int max_states,
		       int max_actions,
		       int tokens_per_node);

//     Delete a brain back to free memory.
int bz_killbrain (bz_brain *brain);

//     block-style learning memory.
bz_block *bz_newblock (bz_brain *brain);

//     chain-style learning memory.
bz_chain *bz_newchain (bz_brain *brain);

//     Given a brain, and a state, and a set of allowed actions, what's
//     this brain choose to do.  (DEPENDS ON RAND() !!!)
long bz_nextaction (
		     bz_brain *brain,
		     int cur_state,
		     char mask[],       // the set of allowed actions
		     int *underflows    // optional underflows (incremented)
		     );


long bz_addtoblock (
		    bz_block *block,
		    int state,
		    int action);

long bz_learnblock (
		   bz_brain *brain,
		   bz_block *block,
		   float tokens_add,
		   float tokens_multiply,
		   int on_empty);   // 

int bz_zeroblock (bz_block *block);
int bz_killblock (bz_block *block);
  
int bz_prettyprint_block (bz_block  *block);  

bz_chain *bz_newchain (bz_brain *brain);

int bz_addtochain (bz_chain *chain, long state, long action);

int bz_learnchain (bz_brain *brain,
		   bz_chain *chain,
		   float add,
		   float multiply,
		   int on_empty);
int bz_zerochain (bz_chain *chain);
int bz_killchain (bz_chain *chain);

//    Internal use only.  Do not depend on this function
//    being stable!  (note the double-underscore)
float bz__random (float max);


  

