//    Parameter file for ball and track learning problem.
//    The ball is on a tippable track; the algorithm needs to
//    learn how to control the tip to move the ball to the desired
//    setpoint on the track.
//
//    But - the algorithm _doesn't know anything about the actual
//    physics of the ball or track_.  It just experiments and learns
//    and hopefully eventually gets it right.
//
//    Values are quantized, and there is NO "velocity" measure, rather the
//    algorithm sees the current and a few prior quantized position states.
//
//     PHYSICAL BALL AND TRACK MODEL (NOT VISIBLE TO ALGORITHM).
//
//    Note that we assume the track servo is "powerful' compared to
//    the weight of the track and the weight of the ball (which we can
//    normalize to 1 Kg, thank you Galileo).  We also ignore
//    rotational inertia, which can be mirrored to linear inertia in
//    the case of an uncurved linear track.
//
//    Values are all in SI units - meters/kilos/seconds/radians, unless noted.
//
//    First, we define the (hidden) physical model, then the
//    quantization that maps this hidden model into a set of states
//    usable by the algorithm.
//
//   timestep size (seconds)
// #define TIMESTEP 0.0333
#define TIMESTEP 0.0333
//
//      Track Parameters (length in meters, tilt in radians)
//
#define TRACKLEN 1.0
#define TRACKANGMIN (-0.2)
#define TRACKANGMAX (0.2)
//
//   The current "real" state of the track, in radians (0 = horizontal)
//   and velocity (in radians/sec)
float track_ang, track_v;
int quantized_track_ang ;
#define INITIAL_TRACK_ANGLE 0.0
#define INITIAL_TRACK_VEL 0.0
#define TRACK_SLEW_RATE 0.50 

#define INITIAL_BALL_X 0.0
#define INITIAL_BALL_V 0.0

//    The current "real" state of the ball, not quantized!
//    (ball_x is 0 to TRACKLEN, ball_v is ball velocity in meters/sec
float ball_x, ball_v;
int quantized_ball_x;
//    Mass of the ball (we normalize to 1.0 Kg right now)
#define BALL_MASS 1.0
//   amount of noise (perturbation) applied to the "real" ball, in Newtons
#define BALL_NOISE 0.0001
//   amount of measurement noise (jitter) applied to ball, each timestep
#define BALL_JITTER 0.001
//   coefficient of restitution - how fast does the ball recoil when it
//   hits the stops at 0 and TRACKLEN?
#define BALL_BOUNCE 0.5
//   switchpoint between static and dynamic friction (meters/sec)
#define BALL_VEL_FRIC_THRESH 0.05
//   coefficient of static friction
#define BALL_STATIC_FRIC 0.05
//   coefficient of dynamic friction
#define BALL_DYN_FRIC 0.02

//
//     QUANTIZING AND MAPPING THE MODEL :  THE LEARNING ALGORITHM SEES
//     THIS STUFF 
//
//   number of ball position quantization states
#define NBALLQ 5
//   number of rail position quantization states
#define NTRACKQ 5
//   number of previous positions visible to the BZERKER algorithm
#define TVIS 1
//   number of total states visible to BZERKER 
#define STATES ((int)(pow((NBALLQ*NTRACKQ),TVIS)))
//   number of possible actions to take - tilt to left, center, or right.
#define ACTIONS 3

//   State memory of the ball and track
int qball_vec [TVIS];
int qtrack_vec [TVIS];

long unsigned quan_state;

//    The following are BZERKER parameters
//   how many tokens per Michie box
#define TOKENS 100
//   how many cycles of the game to run
#define REPEATS   500
//   batch size (statistics gathering only)
#define BATCHSIZE 10000
//   reward params for the boxes (default at least)
#define WIN_ADD 1.0
#define WIN_MUL 1.0
#define LOSE_ADD (-1.0)
#define LOSE_MUL 1.0
#define DRAW_ADD 0.01
#define DRAW_MUL 1.0
#define PRINT_EACHGAME 0

//     The reward parameters - how close is the _real_ ball to the _real_
//     setpoint?
float cur_reward;
//        the setpoint
#define BALL_SETPOINT 0.50
//        The maximum (default) reward
#define BALL_MAXREWARD 1
//        The maximum (default) punishment
#define BALL_MAXPUNISH (-1)
//        How much taken off per meter of error?
#define BALL_ABS_TAPER 2
//        How much taken off per meter error SQUARED?
#define BALL_SQUARED_TAPER 4



