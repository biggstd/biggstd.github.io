# Gillespie Stochastic Simulations

Here I implement the algorithm described in *Exact Stochastic Simulation of Coupled Chemical Reactions* by Danlel T. Gillespie. It is a seminal paper and well worth the read, a copy can be found online [here](https://www.caam.rice.edu/~cox/gillespie.pdf). Much of the code below borrows notation from the paper.

Following the paper, for testing our algorithm I use [exponential decay](https://en.wikipedia.org/wiki/Exponential_decay). Such processes are ubiquitous in nature, so the example is worthwhile. Additionally we can numerically solve many features of interest in this system, and
compare those features to our simulation results.


```python
import numpy as np
from functools import partial
from manim import *
rng = np.random.default_rng()

import matplotlib.pyplot as plt
%matplotlib inline
```


## Exponential Decay

First I construct and plot the output of a parameterized exponential decay function.

There are some other features we can calculate for exponential decay.
Of interest here is the root-mean-squared error, which in this case is the same as the standard deviation:

$$
\sigma = [X_0 e^{-ct}(1 - e^{-ct})]^{\frac{1}{2}}
$$

$\pm$ 3σ should contain about 97% of our data from our stochastic simulation of this same process.


```python
def exp_decay(X, c, t):
    """
    X: initial number of a species
    c: exponential decay constant
    t: a give time
    """
    return X * np.exp(-c * t)


def rms_exp(X, c, t):
    return (X * np.exp(-c * t) * (1 - np.exp(-c * t)))**(0.5)


time_range = np.linspace(0, 10, 1000)
x_decay = exp_decay(100, 0.5, time_range)
rms = rms_exp(100, 0.5, time_range)
# plus minus 6σ should contain about 97% of our data.
upper = x_decay + rms*3
lower = x_decay - rms*3

fig, ax = plt.subplots(figsize=(7, 3))
ax.set(ylim=(0, 110), xlim=(0, time_range[-1]), 
       xlabel="Time", ylabel="[X]", title='Exponential Decay with $\pm 3\sigma$')
ax.plot(time_range, x_decay);
ax.fill_between(time_range, upper, lower, facecolor='grey', alpha=0.5);
```


<image src="./assets/gillespie/output_4_0.png"  width="500" >output image</image>


## Gillespie Algorithm

Here I implement a version of Gillespie's algorithm. I wrote this code when I was still learing programming, and my goal was more to follow the steps outlined in the paper. It works fine, but certainly is not the most robust thing ever written. For example, it just crashes if there are
more iterations that possible reactions. That said, the code is fairly clear to read and faithful to the original paper.


```python
class Gillespie:
    
    def __init__(self, X, C, delta_X, H):
        self.X = np.asarray(X, dtype=int)
        self.C = np.asarray(C)
        self.delta_X = np.asarray(delta_X, dtype=int)
        self.H = H

    @staticmethod
    def calc_av(X, H, C):
        """Calculate the Av value.
        The number of available permutations multiplied be the rates."""
        collisions = np.array([h(X) for h in H])
        return collisions * C
    
    @staticmethod
    def calc_tau(a0, rv):
        """Calculate the Tau value, which is the probable length of time
        before any given simulated reaction occurs."""
        return (1 / a0) * np.log(1 / rv)
    
    @staticmethod
    def calc_mu(av, a0, rv):
        """Calculate the mu value.
        
        :param Av_vals:
            The possible reaction permutations * their rates. Given
            as a numpy array.
        :param Av_sum:
            The sum of the `Av_vals` array. This sum is used in
            multiple places, so it is not calculated within this
            function.
        :param random_value:
            A randomly generated value between zero and one.
        :returns mu:
            The index corresponding to the reaction that should be
            simulated.
            
        Essentially we generate blocks of value on a number line
        from zero to one. A random number determines where on this
        line a reaction "occurs".
            [================================================]
            0.0                  *                          1.0
                                  A random point.
                                  
        We cast the possible reactions to this scale by multiplying
        the random value by the sum of Av values. Such casting is
        done by chunks.
            [================================================]
            [=Chunk 1=][======Chunk 2======][=====Chunk 3====]
            
        The sums of these chunks are examined, and when the sum is 
        found to be greater than the randomly cast point defined above, 
        the corresponding reaction is simulated.
        
        ..warning::
            A different random value should be used for `calc_mu()` and
            `calc_tau()`.
        """
        return np.argwhere(np.cumsum(av) > (rv * a0)).flatten()[0]
    
    def simulate(self, max_iter=1000, max_time=1000):
        """Runs a stochastic simulation based on the input provided. If reactants
        are consumed before max_iter this will not work."""
        curr_time = 0
        curr_iter = 0
        
        t_out = np.empty(max_iter)
        x_out = np.empty((max_iter, self.X.shape[0]))
        
        t_out[0] = curr_time
        x_out[0] = self.X.copy()
        
        while (curr_iter < max_iter) and (curr_time < max_time):
            av = self.calc_av(self.X, self.H, self.C)
            # print(av)
            a0 = np.sum(av)
            # print(a0)
            rv1, rv2 = rng.random(2)
            tau = self.calc_tau(a0, rv1)
            # print(tau)
            mu = self.calc_mu(av, a0, rv2)
            self.X += self.delta_X[mu]
            curr_time += tau
            t_out[curr_iter] = curr_time
            x_out[curr_iter] = self.X
            curr_iter += 1
                        
        return t_out, x_out
```

### Test the Stochastic Simulation

Now I run a test, and compare the results to the continuous function.


```python
%%time

exp_decay_params = dict(
    X=[100], # A list of counts of species.
    C=[0.5], # A list of reaction rates.
    delta_X=[[-1]], # The result of a reaction, indexed the same as its rate.
    H=[lambda x: x[0]] # The number of possible reactions, indexed the same as its rate.
)

gill = Gillespie(**exp_decay_params)
times, x_vals = gill.simulate(100)

fig, ax = plt.subplots(figsize=(7, 3))
ax.set(ylim=(0, 110), xlim=(0, time_range[-1]), 
       title='Simulation run compared to exponential decay')
ax.plot(time_range, x_decay, linewidth=0.75)
ax.fill_between(time_range, upper, lower, facecolor='grey', alpha=0.5);
ax.scatter(times, x_vals.flatten(), s=2, color='k');
```

    CPU times: total: 15.6 ms
    Wall time: 12 ms
    


<image src="./assets/gillespie/output_8_1.png"  width="500" >output image</image>
    


Plot several hundered runs to see how our points fall.


```python
%%time
exp_runs = [Gillespie(**exp_decay_params).simulate(100) for i in range(200)]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.set(ylim=(0, 110), xlim=(0, 10), title='Gillespe Algorithm and Exponential Decay')
ax.plot(time_range, x_decay)
ax.fill_between(time_range, upper, lower, facecolor='grey', alpha=0.5);
for run in exp_runs:
    t, x = run
    ax.scatter(t, x, s=.1, alpha=0.75, color='k')
```

    CPU times: total: 500 ms
    Wall time: 503 ms
    



<image src="./assets/gillespie/output_10_1.png"  width="500" >output image</image>


## Self-equilibrating Reactions


```python
self_eq_params = dict(
    X=[3000],
    C=[5, 0.005],
    delta_X=[[1], [-2]],
    H=[
        lambda x: x[0], 
        lambda x: x[0] * (x[0] - 1) / 2,
      ]
)

gill = Gillespie(**self_eq_params)
times, x_vals = gill.simulate(100000)

fig, ax = plt.subplots(figsize=(7, 3.5))
for x in x_vals.T:
    ax.scatter(times, x, marker=".", s=.1, alpha=0.1)
plt.axhline(y=1000, c='k')
```




    <matplotlib.lines.Line2D at 0x1f29176bdd0>




<image src="./assets/gillespie/output_12_1.png"  width="500" >output image</image>   
    


## Lotka Reaction

Originally developed to model predator-prey interactions.


```python
lotka_params = dict(
    X=[1000, 1000],
    C=[10, 0.01, 10],
    delta_X=[[1, 0], [-1, 1], [0, -1]],
    H=[
        lambda x: x[0], 
        lambda x: x[0] * x[1],
        lambda x: x[1]
      ]
)

gill = Gillespie(**lotka_params)
times, x_vals = gill.simulate(90000)

fig, ax = plt.subplots(figsize=(7, 3.5))
for x in x_vals.T:
    ax.scatter(times, x, marker=".", s=.1, alpha=0.1)
```


<image src="./assets/gillespie/output_14_0.png"  width="500" >output image</image>   



What I find interesting in this system is the orbital phase representation.


```python
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.scatter(x_vals.T[0], x_vals.T[1], marker=".", s=.1, alpha=0.6)
```




    <matplotlib.collections.PathCollection at 0x1f281955d10>




<image src="./assets/gillespie/output_16_1.png"  width="500" >output image</image>   
    


## Making a Video with Manim

I had fun with this project. In my experience seminal papers often work much harder to teach and convince the reader than follow up work attempts. The paper by Gillespie is no exception, and lays out the process I implemented here in great detail.

As I was preparing this notebook for publication I wanted a movie to go along with it. Here I
tried to make a short video highlighting the similarity between the stochastic simulation and the continuous function results.


```python
%%manim -qh -o gillespie.mp4 --format mp4 --media_dir manim_output --disable_caching ExpDecayPlot

exp_runs = [Gillespie(**exp_decay_params).simulate(100) for i in range(20)]

class ExpDecayPlot(Scene):
    def construct(self):

        axes = Axes(x_range=[0, 10, 1], y_range=[0, 100, 10],tips=False)
        self.play(FadeIn(axes))
        
        axes_labels = axes.get_axis_labels(Tex("X"), Text("time"))
        self.play(Write(axes_labels))
        
        exp_func = partial(exp_decay, X=100, c=0.5)
        exp_plot  = axes.plot(lambda t: exp_func(t=t), color=BLUE)
        exp_plot_text = Tex("exponential decay function")
        self.play(Write(exp_plot_text))
        self.play(Create(exp_plot))
        self.play(FadeOut(exp_plot_text))
        self.wait(1)
        
        rms_func = partial(rms_exp, X=100, c=0.5)
        lower_rms_bound  = axes.plot(lambda t: exp_func(t=t) - rms_func(t=t), stroke_opacity=0.5)
        upper_rms_bound  = axes.plot(lambda t: exp_func(t=t) + rms_func(t=t), stroke_opacity=0.5)
        area = axes.get_area(lower_rms_bound, [0, 10], bounded_graph=upper_rms_bound, 
                           color=GREY, opacity=0.5)
        
        lower_rms_bound_text = Tex("lower rms bound")
        self.play(Write(lower_rms_bound_text))
        self.play(Create(lower_rms_bound))
        self.play(FadeOut(lower_rms_bound_text))
        self.wait(1)
        
        upper_rms_bound_text = Tex("upper rms bound")
        self.play(Write(upper_rms_bound_text))
        self.play(Create(upper_rms_bound))
        self.play(FadeOut(upper_rms_bound_text))
        self.wait(1)
        
        rms_bound_text = Tex("95\% ci")
        self.play(Write(rms_bound_text))
        self.play(FadeIn(area))
        self.play(FadeOut(rms_bound_text))
        self.wait(1)
        
        stoch_sim_text = Tex("stochastic simulation")
        self.play(Write(stoch_sim_text))
        gill = Gillespie(**exp_decay_params)
        times, x_vals = gill.simulate(100)
        
        sim_dots = []
        for x, y in zip(times, x_vals.flatten()):
            dot = Dot(axes.c2p(x, y), color=ORANGE, radius=0.02)
            sim_dots.append(dot)

        self.play(ChangeSpeed(Succession(*[FadeIn(d) for d in sim_dots], lag_ratio=0.95),
                  speedinfo={0.0: 20, 0.9: 20, 1.0: 10}))
        self.wait(3)
        self.play(FadeOut(stoch_sim_text))

        multi_stoch_sim_text = Tex("multiple stochastic simulations")
        self.play(Write(multi_stoch_sim_text))
        
        run_dots = []
        for times, x_vals in exp_runs:
            for x, y in zip(times, x_vals.flatten()):
                dot = Dot(axes.c2p(x, y), color=ORANGE, radius=0.02, fill_opacity=0.5)
                run_dots.append(dot)
                
        self.play(FadeIn(*run_dots))
        self.wait(1)
        self.play(FadeOut(multi_stoch_sim_text))
        self.wait(3)
```

                                                                                                                           

<video src="./assets/gillespie/gillespie.mp4" controls  width="500" >
Your browser does not support the <code>video</code> element.</video>


