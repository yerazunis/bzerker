# bzerker
Machine Learning by State Exploration - An implementation of 
Michie's BOXES algorithm, popularized by Martin Gardner 
in March 1962 and used by Fred Saberhagen in the story 
"Without A Thought" in 1963.  

Very simply, BOXES creates a set of STATEs that map onto the 
visible attributes of the system being interacted with, and selects 
one of the ACTIONS permitted in that STATE according to a 
statistical analysis of how well things went the last time that 
STATE / ACTION pairing was selected.   The algorithm then 
is taught that, in this instance, that series of actions 
led to a certain desirability (or undesirability) of result, 
simply by increasing the probability of selecting a desirable 
ACTION and decreasing the probability of an undesirable ACTION. 

STATEs are utterly "blind" - any way to map your system or physical
configuation into unique numbered states is perfectly valid.  
Similarly, ACTIONs are "blind" - any numbering of possible actions
to take, starting at 0, will work.   Right now, sparse ACTIONS are 
permitted (via the MASK vector) but STATEs are still dense.

Because of the extreme simplicity of the algorithm, it is wicked 
fast to run any particular instance of STATE and get
an ACTION back, but overall, it's a bit slow to learn- for 
example, to learn how to play generalized tic-tac-toe to 
"95% perfection" (that being a forced draw), about a million 
games must be run.   However, this takes only about four seconds on 
an old Macbookwith a 2.4 GHz i5 procesor, so it's not even
close to crazy.

This speed makes it quit useful in certain machine learning 
situations such as so-called "model based reinforcement
learning", where the actual physical system is explored and
used to generate a model; the model is then run at tens to 
millions of times faster than realtime to generate a plausibe
strategy (a "policy", in ML-speak), and then, hopefully, the 
generated policy can control the physical system "well enough"
to complete the learning directly.

CAUTION: choice of parameters is important; for example, if one only 
allocates 10 units of liklihood as the starting configuation for 
each possible action given a particular state of a tic-tac-toe 
game; "gambler's ruin" dominates the system (i.e. ten losses for 
each action in each state mean _no_ action is statistically "good" 
and so to avoid this kind of underflow, we reset such states back 
to the initial loading.  However, this means that statistically valid
models don't survive and we never actually learn (or converge)

Going with 100 units of liklihood, we learn in about 1 million 
games (and take about four seconds), and with 1000 units of 
liklihood, it takes about 20 times more games (about 10 million 
games), and the time required goes from 4 seconds to about 100 
seconds.

Similarly, unrewarded strategies don't get any gain.  For example,
tic-tac-toe, with the standard (not generalized) rules is a
forced draw between two perfect players.... but if 'draw' is
considered a loss to player 2, then it's a forced win for player
1 and this is reflected in the results for trying to "learn"
tic-tac-toe with a 0 reward for draw.  The usual contest-style
point system (1 point for win, 0 point for lose, 1/2 point for
draw) gives the expected result, but slowly as losing strategies
are never "punished".   But, +1 for win, -1 for lose, and +0,01
for draw converges quickly.

Realize that your reward function does not have to be zero-sum.
Empirically, it seems that learning can be faster if it is NOT
zero-sum; I don't know why this is.

So, it's not perfect, you have to think carefully about your 
reward function, but it _is_ fast.   

Enjoy.

        - Bill Yerazunis
