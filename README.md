# Tyler Biggs' Website

Welcome, my name is Tyler, I am a chem/bio-informatics research scientist and statistician. 

I obatined my Ph.D. in synthetic organic chemistry at Washington State University. In my post-doctoral
research I transitioned into programming, statistics and machine-learning. I occasionally do freelance
statistical consulting.

My non-science hobbies inlcude houseplants, terrariums and a bit of D&D.

This site is a small collection of data science themed projects that I found interesting, and I hope you do as well.

[LinkedIn](https://www.linkedin.com/in/tylerbiggs/)
[General Resume](./docs/Biggs_Tyler_Resume_Jan_2024_Public.pdf)


## Projects

> For the things we have to learn before we can do them, we learn by doing them.
> -Aristotle

### [***Learning Programming with Conway's Game of Life***](./docs/raytrace_gol)

I began work on a short course or module designed to introduce programming for data science to 
graduate students. Preparing material for students with such a wide range of potential experiences is challenging,
and as the development of the course was cancelled, I never was able to polish the course to any degree. That said, 
the capstone project for the students produces some lovely ray-traced images, and the method we use to permute each 
step of the game using a convolution is a good step from a programming point of view to a mathematical and scientific one.

<video src="/docs/assets/gol_HDr.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>



### [***Exploring Stochastic Simulations with a Seminal Paper***](./docs/Gillespie_Stochastic_Simulations)

Here I implement the algorithm described in *Exact Stochastic Simulation of Coupled Chemical Reactions* by Danlel T. Gillespie. It is a seminal paper and well worth the read, a copy can be found online [here](https://www.caam.rice.edu/~cox/gillespie.pdf). Much of the code below borrows notation from the paper. Following the paper, for testing our algorithm I use [exponential decay](https://en.wikipedia.org/wiki/Exponential_decay). Such processes are ubiquitous in nature, so the example is worthwhile. Additionally we can numerically solve many features of interest in this system, and compare those features to our simulation results.

<video src="/docs/assets/gillespie/gillespie.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>


### [***Reviewing old Quantum Mechanics Notes***](./docs/quantum_tunneling)

I was visiting my parents for the holidays and found some old class notes from introductory quantum mechanics, a class that only began to make sense in retrospect. Since then, I have gained some programming skills, so I set out to see if I could numerically solve some simple 1-dimensional systems. It turned out to be successful, and using Manim to make the animation shine was fun. There are several other versions of this online, but I found this one from Dot Physics to be the most informative. Even still, for many of the examples I saw, the code reads differently than the math a student encounters, which is fine for the program but not for the student. Here, I implement a Gaussian wave and its interaction with a barrier to draw a clear link between the math used and the code employed.

<video src="/docs/assets/quantum_tunneling/quantum_tunneling.mp4" controls  width="500" >Your browser does not support the <code>video</code> element.</video>