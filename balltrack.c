//   Ball-and-tipping-track simulator test for the bzerker library
//   implementing Mitchie's BOXES algorithm
//   Copyright 2017-2018 W.S.Yerazunis, released under the GPL V2 or later
//
//
//  Learn to balance a ball on a tipping track with reinforcement learning
//
//  Data structures:
//     The ball: has a position and a velocity, which are NOT directly
//               seen by the learning process.  The ball is actually
//               represented by floats, and may have a frictional
//               term added - even a nonlinear frictional term.
//     The track: also has an angular position and velocity, which are NOT
//               directly seen by the learning process.  The servo that
//               tips the track is noisy and laggy and slow (these
//               parameters are variable.
//  Actions Out: integer (always allocated) - representing the tip servo's
//               target setpoint (midpoint is roughly horizontal; positions can
//               be varied in the code).  Note that the servo takes a
//               long time to actually arrive at it's setpoint.
//  States:  We quantize the state of the ball on the beam into Nb slots
//               of position for the ball, and Nt slots for the track angle.
//               We do NOT try to model the velocity "in problem state",
//               instead we show the last T states to the learning algorithm
//               and let the learning algorithm infer the laws of motion
//               by reinforcement learning.
//
//               A number of different experiments are possible while
//               staying in the computationally feasible domain for
//               reinforcement learning (on the order of 10K states for
//               a single laptop).
//               Nb = Nt = 10, T = 2 : quantize ball and track to 10 states,
//                    let the algorithm see the current and single previous
//                    state of the ball and track.
//               Nb = Nt = 5, T = 3 : quantize the ball and track to 5
//                    states, reveal the current and two prior states
//               Nb = Nt = 3, T = 4 : quantize ball and track to 3 states,
//                    reveal the current and three prior states.
//
//   Other encodings are possible; that's part of the experiment.
//   With generalized tic-tac-toe, there were 3^9 = 19,683 possible
//   states (not all were reachable, such as states where X made 8 moves
//   and 0 made 1); a similar situation exists here given the physical
//   model should (hopefully) not produce unphysical results.
//

#include "bzerker.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "balltrack.h"


bz_brain *brain1;   // our brain
bz_chain *chain1;   // the chain of actions taken
int *gamblers_ruins;

////////////////////////////////////////////////////////////////
//
//   Some global vars (yeah, global vars are a tool of Satan.  But
//   reinforcement learning is a Dark Art to start with.

int track_cmd;

///////////////////////////////////////////////////////////////////
//   Some function definitions.

//   set up initial conditions
void init_physics () {
  //  The actual physics model init:
  track_ang = INITIAL_TRACK_ANGLE ;  // in radians
  track_v = INITIAL_TRACK_VEL;    //  radians/sec
  ball_x = INITIAL_BALL_X ;       // in meters
  ball_v = INITIAL_BALL_V ;       //   meters/sec
  track_cmd = (ACTIONS - 1) / 2;   //  pick an initial track command
  int i;
  //   The observations we have:  
  for (i = 0; i < TVIS; i++) {
    qball_vec [i] = 0;
    qtrack_vec [1] = 0;
  }
}

//   Move the track according to one track command (range 0
//   to ACTIONS-1) of an RC-servo-driven track.
//
//   Note that the quantization of command ACTIONS (the track angle
//   setpoint) is independent of the quantization of position visible
//   to the BZERKER algorithm, and need not contain a "stable" state
//   such as horizontal!
//
//   Note also that the track has finite and limited speed, as
//   well as limits on maximum tilt (look in balltrack.h)
//
void move_track_one_timestep (int track_cmd) {
  float track_angrange, track_setpoint;
  track_angrange = TRACKANGMAX - TRACKANGMIN;
  //   translate track_cmd (an int, 0 to ACTIONS-1) into radians
  track_setpoint = (track_cmd * (track_angrange / ACTIONS)) + TRACKANGMIN;
  //  are we close enough to just move there?
  if (abs (track_setpoint - track_ang) < (TRACK_SLEW_RATE / TIMESTEP)) {
    track_ang = track_setpoint;
  }
  //  No, we can't get to setpoint in one step.  We should take one
  //  step worth closer
  else {
    if (track_setpoint > track_ang) {
      track_ang += (TRACK_SLEW_RATE / TIMESTEP);
    } else {
      track_ang -= (TRACK_SLEW_RATE / TIMESTEP);
    }
  }
  return;
}

//   After the track moves, let the ball move
void move_ball_one_timestep () {
  float ballforce;
  ballforce = 0;
  /////////////////////////////////////////
  //     Part 1: account for all forces on the ball
  //
  //  what force do we get from the inclined track
  ballforce += BALL_MASS * sin (track_ang);
  
  //  apply friction to the ball
  if (ball_v < BALL_VEL_FRIC_THRESH) {
    //   static friction case
    ballforce += (BALL_MASS * BALL_STATIC_FRIC)
      * (ball_v > 0.00) ? -1 : +1;    //  friction switches direction
  } else {
    //   dynamic friction case
    ballforce += (BALL_MASS * BALL_DYN_FRIC)
      * (ball_v > 0.00) ? -1 : +1;    //  friction switches direction!
  }
  ///////////////////////////////////////////////
  //    Part 2:  Integrate.   Force = mass * accelleration
  //    so accelleration = force/mass, delta-v = accel * timestep
  ball_v = ball_v + ((ballforce / BALL_MASS) * TIMESTEP);
  //    now integrate velocity to get position
  ball_x = ball_x + (ball_v * TIMESTEP);
  /////////////////////////////////////////////
  //    Now the nonlinearities and noises and bumpers at the end
  if (ball_x < 0) {
    //  Ball bounces off the x=0 bumper!
    //    yes, this isn't exactly right; should break the motion into
    //    two phases, pre-impact and post-impact, but for small timesteps
    //    this is close enough.
    ball_x = -(ball_x * BALL_BOUNCE);
    ball_v = -(ball_v * BALL_BOUNCE);
  }
  ////    Ball bounces off the upper bumper (bumper at TRACKLEN)   
  if (ball_x > TRACKLEN) {
    ball_x =  TRACKLEN - (ball_x * BALL_BOUNCE);
    ball_v = -(ball_v * BALL_BOUNCE);
  }
}

///    Set the quantized values needed for the algorithm's
//     state inputs.   Note that these are normalized to
//     be integers in the 0 to N range, not floats, not negative
//
//     Conveniently, we never have to quantize velocity, because
//     we don't give velocity to the algorithm, just time series
//     of position.  Ain't that cool!  :-)
void set_quantized_values () {
  //  Track angle
  quantized_track_ang = (int) ((track_ang - TRACKANGMIN) * NTRACKQ)
    / (TRACKANGMAX - TRACKANGMIN);  
  //  Ball position
  quantized_ball_x = (int) ((ball_x ) *  NBALLQ) / (TRACKLEN);
}

void set_quantized_state_queues () {
  // slide the state vectors down one step
  int i;
  for (i = 0; i < TVIS-1; i++) {
    qball_vec[i] = qball_vec[i+1];
    qtrack_vec[i] = qtrack_vec[i+1];
  }
  //   and stuff the newest positions in at the end.
  qball_vec[TVIS-1] = quantized_ball_x;
  qtrack_vec[TVIS-1] = quantized_track_ang;
}

//      Turn our state memory into a BZERKER state
//
//      This is equivalent to encoding qtrack_vec and qball_vec as an
//      integer in base NBALLQ & NTRACKQ (interleaved).
void que_to_quan_state () {
  int ti;         //  which timeslice
  long unsigned bstate;     //  bzerker state accumulator
  long unsigned bmax;       //  maximum base multipilier
  bstate = 0;
  bmax = 1;
  for (ti = 0; ti < TVIS; ti++){
    bstate += qball_vec[ti] * bmax;
    bmax = bmax * NBALLQ;
    bstate += qtrack_vec[ti] * bmax;
    bmax = bmax * NTRACKQ;
  }
  quan_state = bstate;
}
  
//    Reward / punishment function

void update_reward () {
  float reward;
  reward = BALL_MAXREWARD;
  int ti;
  float abs_ball_error;
  abs_ball_error = fabs (ball_x - BALL_SETPOINT);
  //   linear taper error
  reward = reward - BALL_ABS_TAPER * abs_ball_error;
  //   quadratic taper error
  reward = reward - BALL_SQUARED_TAPER * abs_ball_error * abs_ball_error;
  cur_reward = reward;
}
				      

main ()
{
  printf ("Starting Ball and Track test - balancing a ball\n");
  bz_init();
  printf ("Important Params:");
  printf ("  Timestep: %f, State History Visible: %d steps.\n", TIMESTEP, TVIS);
  printf ("  Quantization:  ball pos %d states,  track ang %d states.\n", NBALLQ, NTRACKQ);
  
  printf (" Initializing the brain.\n");
  bz_brain *brain1;
  brain1 = bz_newbrain (BZ_BRAIN_QUANTIZED, STATES, ACTIONS, TOKENS);
  printf ("Got brains! pointer is %lx\n",(long) brain1);
  bz_chain *chain1;
  chain1 = bz_newchain (brain1);
  
  printf (" I will run %d steps of balancing, and report every %d steps.\n",
	  REPEATS, BATCHSIZE);
  long reps, i, j, action, batch;

  //   The big loop, where we repeatedly:
  //   (0 - initialize only)
  //   1)  move the track
  //   2)  move the ball
  //   3)  quantize the track and ball
  //   4)  update the current-and-prior state queues
  //   5)  update the reward
  //   6)  train the brain (if we should)
  //   7)  ask the brain what to do next
  //   .... for REPEAT reps.

  //   initialize physics and stuff.
  init_physics ();
  
  //   Loop for REPEAT reps
  for (reps = 0; reps < REPEATS; reps++){
    //   1)  move the track
    move_track_one_timestep (track_cmd);   //  GROT FROM WHERE DO WE GET THIS?
    //   2)  move the ball
    move_ball_one_timestep ();
    //   3)  quantize the track and ball
    set_quantized_values ();
    //   4)  update the current-and-prior state queues
    set_quantized_state_queues ();
    //   5)  update the reward
    update_reward ();
    //   6)  train the brain (if we should)
    que_to_quan_state();
    bz_addtochain (chain1, quan_state, track_cmd);    // remember what we did
    bz_truncatechain (chain1, TVIS); //  .... but only for our memory step length
    //       ... only train if we've actually filled the queues
    if (reps > TVIS) {
      bz_learnchain (brain1, chain1, cur_reward, 1.0, NULL);
    }
    //   7)  ask the brain what to do next
    track_cmd = bz_nextaction (brain1, quan_state, NULL, NULL, NULL);
    //   8)   record statistics and output trace 
    printf ("Ang: %f  BallX: %f  BallV: %f  ErrDist: %f  Score: %f\n",
	    track_ang, ball_x, ball_v, BALL_SETPOINT - ball_x, cur_reward); 
  }  
}
