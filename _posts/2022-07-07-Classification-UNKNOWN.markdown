---
layout: post
comments: true
mathjax: true
title: “Classification UNKNOWN”
excerpt: “Classification UNKNOWN”
date: 2022-07-07 08:00:00
---


### Gaussian distribution (Quick review)
We define a function $$ y = g(x) $$ to map input $$x$$ to $$y$$. In statistic, we use a stochastic model to define a probability distribution for such relationship.  For example, a 3.8 GPA student can earn an average of $60K salary with a variance ($$\sigma^2$$) of $10K.

<div class="imgcap">
<img src="/assets/ml/gpa.png" style="border:none;width:40%">
</div>

$$ p(Salary=x|GPA=3.8)  \quad \text{ (a Gaussian distribution with } \mu = $60K \text{ and } \sigma^2=$10k)$$

**Probability density function (PDF)** 

In the following diagram, $$p(X=x)$$ follows a gaussian distribution: 
<div class="imgcap">
<img src="/assets/gm/g0.png" style="border:none;width:60%">
</div>

$$
PDF = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x - \mu)^{2}/2\sigma^{2} } 
$$

In a Gaussian distribution, 68% of data is within 1 $$\sigma $$ from the $$ \mu $$ and 95% of data is within 2 $$  \sigma $$. 

We can sample data based on the probability distribution. The notation to sample data from a distribution $$\mathcal{N}$$ is:

$$
x \sim \mathcal{N}{\left(
\mu 
,
\sigma^2
\right)}
$$


> In many real world examples, data follows a gaussian distribution. 
