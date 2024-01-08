---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

Welcome, my name is Tyler. I began programming during my doctoral research in synthetic organic chemistry,
and have now transitioned into chem/bioinformatics.

+ [LinkedIn](https://www.linkedin.com/in/tylerbiggs/)  
+ [General Resume](https://github.com/biggstd/biggstd.github.io/blob/master/docs/Biggs_Tyler_Resume_Jan_2024.pdf)

This site is a small collection of 'data science' projects that I found interesting, and I hope you do as well.

> For the things we have to learn before we can do them, we learn by doing them.
> -Aristotle

## [Learning Programming with Conway's Game of Life](./projects/raytrace_gol)

Here I present a capstone project for an introductory programming course for data science. It produces 
some lovely ray-traced images, and the method we use to permute each step of the game using a convolution 
is a good step from a programming point of view to a mathematical and scientific one.

<video src="/projects/assets/gol/gol_HDr.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>


## [Exploring Stochastic Simulations with a Seminal Paper](./projects/Gillespie_Stochastic_Simulations)

Here I implement the algorithm described in *Exact Stochastic Simulation of Coupled Chemical Reactions* by Danlel T. Gillespie. It is a seminal paper and well worth the read, a copy can be found online [here](https://www.caam.rice.edu/~cox/gillespie.pdf). Much of the code below borrows notation from the paper. Following the paper, for testing our algorithm I use [exponential decay](https://en.wikipedia.org/wiki/Exponential_decay). Such processes are ubiquitous in nature, so the example is worthwhile. Additionally we can numerically solve many features of interest in this system, and compare those features to our simulation results.

<video src="/projects/assets/gillespie/gillespie.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>


## [Reviewing old Quantum Mechanics Notes](./projects/quantum_tunneling)

I was visiting my parents for the holidays and found some old class notes from introductory quantum mechanics, a class that only began to make sense in retrospect. Since then, I have gained some programming skills, so I set out to see if I could numerically solve some simple 1-dimensional systems. It turned out to be successful, and using Manim to make the animation shine was fun. There are several other versions of this online, but I found this one from Dot Physics to be the most informative. Even still, for many of the examples I saw, the code reads differently than the math a student encounters, which is fine for the program but not for the student. Here, I implement a Gaussian wave and its interaction with a barrier to draw a clear link between the math used and the code employed.

<video src="/projects/assets/quantum_tunneling/quantum_tunneling.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>