//   Bzerker - an implementation of Michie's 1960 BOXES algorithm for AI.
//   as popularized by Martin Gardner in Scientific American in March 1962 and
//   then used as a plot device by Fred Saberhagen in his Berserker story
//   "Without A Thought" in 1963, which is where I first read of it sometime
//   around 1968.  Note that "Berserker" in this context is the trademark
//   of Fred Saberhagen's literary estate, so other than the references
//   referring specifically to Saberhagen's authorship, we will not use
//   the word in this work.  Instead, we will use Bzerker (pronounced
//   BEE-zerker, like an african bee), as an homage and reminder that
//   sometimes a great mind does math, and sometimes a great mind writes
//   stories, and sometimes you just need both.
//   
//   Copyright W. S. Yerazunis 2017-2018; released under the GPL version
//   2 or later version.
//
//   Version 0.01 - Just Testing Some Ideas I've Had Since I Was Twelve
//     And Read A Really Good Story About Giant Doomsday Machines In Space.
//
//
//   The Algorithm (expressed as a game with 1 or more turns):
//      
//    INIT: for every possible situation in a game:
//          create a "box";
//          for each possible move in that game situation:
//             fill the box with ten (or so) tokens for each one of the
//             legal moves that can be made in that game situation (the
//             tokens say what move they represent);
//
//    STEP: find the box that corresponds to the current game situation;
//          from that box, randomly (*) choose one of the tokens;
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
//   pencil and paper (and later a simulation in FORTRAN); Martin
//   Gardner's implementation used M&M candy, and Fred Saberhagen's
//   story used toy plastic beads.   
//
//   (*) the original Saberhagen Berserkers used "a block of some long-
//   lived radioisotope" to generate the random numbers required to
//   choose "randomly".  We have no such luxury, and must make do with
//   pseudorandom generators.
//
//   Note that this algorithm does NOT use backpropagation, merely end-to-end
//   acceptability for an enumerated set of moves.   Note also that in
//   the limit of a very large number of training passes and single-turn
//   games, the algorithm convergs to the K-nearest-neighbor answer.
//
//   For convenience and disambiguation, all user-recommended bzerker names
//   start with "bz_".   Internal-use-only names (internal because they may
//   not sanitize their arguments very well, or because their API is subject
//   to change) will start with "bz__" (two underscores).  The idea here is
//   that if you stick with "bz_" things, you are unlikely to ever have to
//   revise your code because the new version of bzerker was incompatible.
//
// EXTENSIONS (maybe now, maybe later, maybe never):
//
//   Mitchie's work depended on the totally discrete nature of the problem
//   being learned; can we extend the algorithm to either nearest-seen
//   or interpolation, on input and/or output?
//
//   How about non-discrete (i.e. continuous) input state variables?  
//
//   How small can we make it?   DSPic-small?  Arduino-small?  Since those
//   embedded platforms rarely have anything higher level than C (not even
//   C++) we'll implement in pure ansi C here.
//
//   For large problems, can we break down the situation (i.e. the state) into
//   a bunch of chunks, each chunk to be treated identically and then the
//   final action taken from the ensemble result?
//
//   The bzerker API also allows multiple simultaneous copies of the
//   algorithm to run simultaneously with different data (and with luck,
//   someday it may even run reentrantly so multicore processing may be
//   eventually possible someday.  But until we make sure it's
//   reentrant, stick with one thread, OK?)
//
// HOW TO USE IT
//
//   First, pop open bzerker.h and look at the declarations there.
//
//   Then, if you like, read one of the bzerker_testN.c programs.
//
//   Then, the basic (recommended) program structure is:
//     initialize the package:
//
//          bz_init();
//
//     then declare (i.e. statically allocate) pointers to one or
//     more "brains" (Michie box arrays).
//
//         *bz_brain mybrain;
//
//     and then initialize the brain structure: call
//
//          mybrain = bz_newbrain_quant (states, actions, tokens);
//
//     to actually create the Michie BOXES brain structure (which is
//     several layers deep and you shouldn't need to dive into unless
//     you want to see how things work inside... which is just fine too!).
//
//       The variables:  "States" is how many discrete
//       states we are quantizing the problem into, "actions" is how many
//       discrete possible actions we can take at this state (which will
//       probably move us to another state) and "tokens" is how many Michie
//       tokens each action gets at the start (tokens are added and removed
//       at each learning cycle; fractional tokens are permitted)
//     
//     Then, to learn a problem, you will need to create a problem simulator
//     that allows your code to execute the series of actions and returns
//     a state and a success/failure/neither flag.  The state is as above,
//     success, failure, and neither flag values indicate to your toplevel
//     code that the training should be "rewarded", "punished" or "unknown,
//     don't do anything yet".  Whatever way you choose to simulate the
//     problem itself is entirely up to you, as is the API to call the
//     simulator.
//
//     Now that you're ready to run one to-the-end cycle of simulation,
//     create a chain of actions; this is where actions are stored so that
//     once the result of a series of actions terminates with a reward/punish
//     situation, the chain can be recalled.  Do this by:
//
//           bz_chain *my_chain;   // execute this once to alloc pointer space.
//
//     and then
//
//           my_chain = bz_newchain (my_brain);   // to actually init storage
//     
//     At this point, you can train the brain by the following procedure.
//     First, ask the brain for an action with bz_next_action:
//
//         action = bz_next_action (*brain, state, *mask, *underflows)
//
//     which returns an integer action.  The variable *brain is a pointer
//     to the brain struct, "state" is an integer state of the system,
//     "*mask" is a *char array of length max_actions, where <=0  entries
//     mark forbidden actions, and >0 entries mark permitted actions.
//     Note, of course, that using \0 as a character in a char string
//     is fraught with peril should you use any function that expects null
//     terminated strings!!!
//
//     *underflows is a pointer to an int; this int gets incremented
//     every time there's fewer than one token in the permitted set of
//     actions (when this happens, all boxes for that state get reinitialized
//     with the initial number of tokens again, and the *underflows counter
//     gets incremented; if you set *underflows to NULL it won't be incremented)
//
//     Once you have an action, you should add the action to the action
//     chain for this training by calling bz_addtochain:
//
//        bz_addtochain (*chain, state, action);
//
//     which will store that action for later training when we know
//     whether this particular set of actions was good or bad.
//
//     You also feed that action to your simulator, which will change
//     state, and possibly return the "reward" or "punish" flags.
//     If neither is returned, then go around the loop again, getting
//     a new action with bz_next_action, adding it to the learnable
//     chain with bz_addtochain, and evaluating it with your simulator.
//
//     Eventually, your simulator will return either a positive result
//     or a negative result; when this happens, it's time to use the
//     chain of actions to train the Michie BOXES brain.  Do this by:
//
//         bz_learnchain (*brain, *chain, tokens_add, tokens_multiply);
//
//     This will change the number of tokens in each of the Michie
//     boxes, increasing the chance that the next time through, the actions
//     that lead to better outcomes have higher probability and the actions
//     with worse outcomes have lower probability.
//
//     Finally, after training the chain, you should deallocate the chain
//     with bz_killchain.  Otherwise, actions will continue to accumulate,
//     and you'll leak memory as well.
//
//         b_killchain (*chain);
//
//     Congratulations; you've now gone through one simulation and learned
//     from it.  Repeat again from bz_newchain() over and over until your
//     brain is close enough to perfect for your needs.
//
//     An exammple:  For tic-tac-toe, there are 3^9 possible boards (we
//     will ignore rotational and mirror symmetries and "unreachable"
//     boards, such as both players winning simultanously.  So the number
//     of board states is 3^9 or 19,683.  The maximum number of actions at
//     each state is nine (nine possible moves for the first move), so
//     actions (== the number of Michie boxes at each state) is also 9.
//
//     We also have to allow for three possible results: winning, losing,
//     and drawing the game.  For each "win", we award (typically) one
//     extra token to each of the moves that won, subtract one plus epsilon
//     token from each of the moves that lost, and award epsilon tokens
//     (typically 0.01 tokens) to each move that ended in a draw game.
//
/////////////////////////////////////////////////////////////////////////
//
//      The typedefs and globals
//
/////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "bzerker.h"

int bz_tracemode = 0;

////////////////////////////////////////////////////////////////////////
//
//      The functions to implement BOXES 
//
////////////////////////////////////////////////////////////////////////

//       initialize?    Nothing here yet.
void bz_init () {
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "Initialization called\n"); 
  }
}

char *bz_version () {
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "Version string requested\n"); 
  }
  return BZ_TRACEPREFIX;
}

char *bz_status () {
    if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "Status called\n"); 
  }

}
bz_brain *bz_newbrain (	     int braintype,
			     int max_states,
			     int max_actions,
			     int tokens_per_node)
{
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "newbrain_quant called\n");
  }
  if (braintype != BZ_BRAIN_QUANTIZED)  {
    fprintf (stderr, "BZERKER - no such brain type.\n");
    exit (1);
  }
  //    Allocate space for this brain's header
  bz_brain *my_brain;
  my_brain = (bz_brain *) (malloc (sizeof (bz_brain)));
  //    Fill in the slots of the brain:
  my_brain->braintype = 0;    //  discrete brain
  my_brain->maxstates = max_states;
  my_brain->maxactions = max_actions;
  my_brain->starting_tokens = tokens_per_node;  //  Used during out-of-token refills
  //    Make the boxes array (which is an array of pointers)
  my_brain->states = (bz_state **) malloc (max_states * sizeof (bz_state *));  
  //    Now fill in the boxes.   This is a dense, discrete brain.
  //     Sparse brains will come later, if ever.
  int istate;
  //    Enumerate each of the allowed states, and put the boxes in each
  for (istate = 0; istate < max_states; istate++) {
    my_brain->states[istate] = (bz_state *) malloc (sizeof (bz_state));
    my_brain->states[istate]->len = max_actions;
    my_brain->states[istate]->actions =
      (float *) malloc(max_actions * sizeof (float));
    int iaction;
    for (iaction = 0; iaction < max_actions; iaction++) {
      my_brain->states[istate]-> actions[iaction] = tokens_per_node;
    }
  }
  return (my_brain);
}

//    free() a brain, including all of the nested boxes, back into
//    free memory....  Because there's pointer chasing involved, we
//    can't just free(brain), but have to free() the stuff from the
//    bottom up.
int bz_killbrain (bz_brain *brain)
{
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "killbrain called\n"); 
  }
  long i;
  for (i = 0; i < brain->maxstates; i++) {
    free (brain->states[i]->actions);
    free (brain->states[i]);
  }
  free (brain);
}

//    Operators on an action "block" - assumption of dense action sets here!!!
//    Instead of a linked list, a block is a fixed length array of
//    the maximum length ever possibly needed (so it's really ~same size as
//    a brain array).
//
//    We may choose to revisit this in the future... it may be better to
//    convert this to a linked list representation.  Or, convert a brain to
//    *this* representation to get rid of all of the pointers-to-pointers.
//    Or implement both, or something.
//
bz_block *bz_newblock (bz_brain *brain)  
{
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "newblock called\n"); 
  }
  bz_block *my_block;
  my_block = (bz_block *) malloc ( sizeof (bz_block));
  my_block->totalcount = 0;
  my_block->brain = brain;
  long maxtokens;
  maxtokens = brain->maxstates * brain->maxactions;
  my_block->tokens = (long *) malloc (maxtokens * sizeof (long));
  return my_block;
}

int bz_zeroblock (bz_block *block) {
  long i;
  for (i = 0; i <(block->brain->maxstates * block->brain->maxactions); i++) {
    block->tokens[i] = 0;
  }
}

//     
int bz_killblock (bz_block *block) {
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "killblock called\n"); 
  }
  free (block->tokens);
  free (block);
}

int bz_prettyprint_block (bz_block *block){
  int i;
  for (i = 0; i < block->brain->maxstates * block->brain->maxactions; i++) {
    if (block->tokens[i] > 0) printf ("condition: %d  count %d \n", i, block->tokens[i]);
  }
}

//
//       The core of the algorithm- given a set of boxes (the "brain"), and
//       a current state, pick an action randomly! 
long bz_next_action (
		     bz_brain *brain,
		     int cur_state,
		     char *mask,
		     int *underflows
		     )
{
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "next_action called\n"); 
  }
  //   Get a random number picking 1 of N actions...
  int myrandom, i;
  float sumup;
  sumup = 0;
  for (i = 0; i < brain->maxactions; i++) {
    //   hidden assumption that mask is "long enough".  Otherwise,
    //   we'll be sucking mud because C doesn't usually check subscripts.
    //
    if (mask == NULL || mask[i] >= 0) {
      sumup += brain->states[cur_state]->actions[i];  // always >= 0
    }
  }
  //fprintf (stderr, "sumup: %f ", sumup);
  //BUG IN MASK SOMEWHERE NEAR HERE???
  if (bz_tracemode) fprintf (stderr, " total tokens: %f ", sumup);
  if (sumup <= 1) {
    if (0) { printf (" UNDERFLOW "); fflush (stdout);};
    if (underflows) { (*underflows)++;}
    for (i = 0; i < brain->maxactions; i++) {
      if (mask == NULL || mask[i] >= 0)
	brain->states[cur_state]->actions[i] = brain->starting_tokens;
    }
  }
  i = 0;
  myrandom = bz__random (sumup);
  for (i = 0; i < brain->maxactions; i++) {
    if (mask == NULL || mask[i] >= 0) {
      myrandom -= brain->states[cur_state] -> actions[i];
    }
    if (myrandom <= 0) return (i);
  }
  
  return i;
}

long bz_addtoblock (
		    bz_block *block,
		    int state,
		    int action)
{
  if (bz_tracemode) fprintf (stderr, "bz_add_action start\n");
  block->totalcount = block->totalcount + 1;
  long offset;
  offset = (state * (block->brain->maxactions)) + action;
  //printf ("bz_add_action offset is %d\n", offset);
  block->tokens[offset] ++;  
}

long bz_learnblock (
		   bz_brain *brain,
		   bz_block *block,
		   float tokens_add,
		   float tokens_multiply,
		   int on_empty)
{
  if (bz_tracemode) {
    fprintf (stderr, "%s%s", BZ_TRACEPREFIX, "learn_block called\n"); 
  }
  if (brain == NULL) {
    fprintf (stderr, "Null brain in learn!  Skipping.");
    return (1);
  }
  if (block == NULL) {
    fprintf (stderr, "Null block in learn!  Skipping.");
    return (1);
  }
  int state, action, seroff;
  for (state = 0; state < brain->maxstates; state++) {
    for (action = 0; action < brain->maxactions; action++) {
      seroff = state * (block->brain->maxactions) + action;
      //  multiply, then add (fractions allowed!)
      //fprintf (stderr, "state: %d  %f -->",
      //	 state, brain->boxes[state]->actions[action]);
      brain->states[state]->actions[action] =
	tokens_add +
	(tokens_multiply * (brain->states[state]->actions[action]));
      //  zero check - no negativity allowed!
      if (brain->states[state]->actions[action] < 0) {
	brain->states[state]->actions[action] = 0;
      } 
    }
  }
  return (0);
}

//    Chains are a different way to do learnable recordings; a block
//    is a fixed-size array but a chain is a linked list.  You're trading
//    off malloc/free time versus time to scan an entire struct the size of
//    the "brain".

bz_chain *bz_newchain (bz_brain *brain){
  bz_chain *mychain;
  mychain = malloc (sizeof (bz_chain));
  mychain->totalcount = 0;
  mychain->chels = NULL;
  return mychain;
}

//   Add an action to the front of the chain for later learning
int bz_addtochain (bz_chain *chain, long state, long action) {
  bz__chel *mychel;
  mychel = malloc (sizeof (bz__chel));
  mychel->state = state;
  mychel->action = action;
  mychel->next = chain->chels;
  chain->chels = mychel;
  chain->totalcount ++;
  return;
}

//    Truncate a chain to be no longer than "count" actions long
//    and return the number of actions dropped
int bz_truncatechain (bz_chain *chain, long count) {
  bz__chel *mychel, *nextchel; // need to to follow the chain
  int icount;
  int idropped = 0;
  mychel = chain->chels;
  //   Initial state- is this a nonempty bz_chain?
  if (mychel == NULL) return (0);
  //   Step thru the chain, counting "count" elements
  for (icount = 1; icount < count; icount++) {
    nextchel = mychel->next;
    if (mychel == NULL) return (0);
    mychel = nextchel;
  }
  //  mychel_now is now pointing to the last good element.  Cut the
  //  chain but hold onto the free end of to-be-deleted elements!
  nextchel = mychel->next;
  mychel->next = NULL;
  //  now mychel is first chel to be deleted.
  mychel = nextchel;   
  while (mychel) {
    nextchel = mychel -> next;
    free (mychel);
    idropped++;
    mychel = nextchel;
  }
  return (idropped);
}

int bz_learnchain (bz_brain *brain, bz_chain *chain,
		   float add, float multiply, int on_empty) {
  bz__chel *thischel;
  thischel = chain->chels;
  while (thischel) {
    brain->states[thischel->state]->actions[thischel->action] =
      add +
      (multiply * (brain->states[thischel->state]->actions[thischel->action]));
    //  zero check - no negativity allowed!
    if (brain->states[thischel->state]->actions[thischel->action] < 0) {
      brain->states[thischel->state]->actions[thischel->action] = 0; 
    }
    thischel = thischel->next;
  }
}

int bz_killchain (bz_chain *chain) {
  bz__chel *mychel, *nextchel;
  mychel = chain->chels;
  while (mychel) {
    nextchel = mychel -> next;
    free (mychel);
    mychel = nextchel;
  }
  free (chain);
}

//
//    Don't call these unless you absolutely have to
int bz__random_init (seed) {
  srandom (seed);
}

float bz__random (float max) {
  float r;
  r = max * (((float)random())/RAND_MAX);
  //printf (" %f", r);
  return r;
}
