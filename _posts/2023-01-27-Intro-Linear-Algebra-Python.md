---
layout: post
comments: true
mathjax: true
priority: 990000
title: “Introduction to Linear Algebra for Applied Machine Learning with Python.”
excerpt: “Introduction to Linear Algebra for Applied Machine Learning with Python.”
date: 2023-01-27 12:00:00
---

**Free resources**:

- **Mathematics for Machine Learning** by Deisenroth, Faisal, and Ong. 1st Ed. [Book link](https://mml-book.github.io/).
- **Introduction to Applied Linear Algebra** by Boyd and Vandenberghe. 1sr Ed. [Book link](http://vmls-book.stanford.edu/)
- **Linear Algebra Ch. in Deep Learning** by Goodfellow, Bengio, and Courville. 1st Ed. [Chapter link](https://www.deeplearningbook.org/contents/linear_algebra.html).
- **Linear Algebra Ch. in Dive into Deep Learning** by Zhang, Lipton, Li, And Smola. [Chapter link](https://d2l.ai/chapter_preliminaries/linear-algebra.html).
- **Prof. Pavel Grinfeld's Linear Algebra Lectures** at Lemma. [Videos link](https://www.lem.ma/books/AIApowDnjlDDQrp-uOZVow/landing).
- **Prof. Gilbert Strang's Linear Algebra Lectures** at MIT. [Videos link](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/).
- **Salman Khan's Linear Algebra Lectures** at Khan Academy. [Videos link](https://www.khanacademy.org/math/linear-algebra).
- **3blue1brown's Linear Algebra Series** at YouTube. [Videos link](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).

**Not-free resources**:

- **Introduction to Linear Algebra** by Gilbert Strang. 5th Ed. [Book link](https://www.amazon.com/Introduction-Linear-Algebra-Gilbert-Strang/dp/0980232775).
- **No Bullshit Guide to Linear Algebra** by Ivan Savov. 2nd Ed. [Book Link](https://www.amazon.com/No-bullshit-guide-linear-algebra/dp/0992001021).

I've consulted all these resources at one point or another. Pavel Grinfeld's lectures are my absolute favorites. Salman Khan's lectures are really good for absolute beginners (they are long though). The famous 3blue1brown series in linear algebra is delightful to watch and to get a solid high-level view of linear algebra.

If you have to pic one book, I'd pic **Boyd's and Vandenberghe's Intro to applied linear algebra**, as it is the most beginner friendly book on linear algebra I've encounter. Every aspect of the notation is clearly explained and pretty much all the key content for applied machine learning is covered. The Linear Algebra Chapter in Goodfellow et al is a nice and concise introduction, but it may require some previous exposure to linear algebra concepts. Deisenroth et all book is probably the best and most comprehensive source for linear algebra for machine learning I've found, although it assumes that you are good at reading math (and at math more generally). Savov's book it's also great for beginners but requires time to digest. Professor Strang lectures are great too but I won't recommend it for absolute beginners.

I'll do my best to keep notation consistent. Nevertheless, learning to adjust to changing or inconsistent notation is a useful skill, since most authors will use their own preferred notation, and everyone seems to think that its/his/her own notation is better.

To make everything more dynamic and practical, I'll introduce bits of Python code to exemplify each mathematical operation (when possible) with `NumPy`, which is the facto standard package for scientific computing in Python.

Finally, keep in mind this is created by a non-mathematician for (mostly) non-mathematicians. I wrote this as if I were talking to myself or a dear friend, which explains why my writing is sometimes conversational and informal.

If you find any mistake in notes feel free to reach me out at pcaceres@wisc.edu and to https://pablocaceres.org/ so I can correct the issue.

# Table of contents

**Note:** _underlined sections_ are the newest sections and/or corrected ones.

**[Preliminary concepts](#preliminary-concepts)**:
- [Sets](#sets)
- [Belonging and inclusion](#belonging-and-inclusion)
- [Set specification](#set-specification)
- [Ordered pairs](#ordered-pairs)
- [Relations](#relations)
- [Functions](#functions)

**[Vectors](#vectors)**:
- [Types of vectors](#types-of-vectors)
    - [Geometric vectors](#geometric-vectors)
    - [Polynomials](#polynomials)
    - [Elements of R](#elements-of-r)
- [Zero vector, unit vector, and sparse vector](#zero-vector-unit-vector-and-sparse-vector)
- [Vector dimensions and coordinate system](#vector-dimensions-and-coordinate-system) 
- [Basic vector operations](#basic-vector-operations)
    - [Vector-vector addition](#vector-vector-addition)
    - [Vector-scalar multiplication](#vector-scalar-multiplication)
    - [Linear combinations of vectors](#linear-combinations-of-vectors)
    - [Vector-vector multiplication: dot product](#vector-vector-multiplication-dot-product)
- [Vector space, span, and subspace](#vector-space-span-and-subspace)
    - [Vector space](#vector-space)
    - [Vector span](#vector-span)
    - [Vector subspaces](#vector-subspaces)
- [Linear dependence and independence](#linear-dependence-and-independence)
- [Vector null space](#vector-null-space)
- [Vector norms](#vector-norms)
    - [Euclidean norm: $L_2$](#euclidean-norm)
    - [Manhattan norm: $L_1$](#manhattan-norm)
    - [Max norm: $L_\infty$](#max-norm)
- [Vector inner product, length, and distance](#vector-inner-product-length-and-distance)
- [Vector angles and orthogonality](#vector-angles-and-orthogonality)
- [Systems of linear equations](#systems-of-linear-equations)

**[Matrices](#matrices)**:

- [Basic matrix operations](#basic-matrix-operations)
    - [Matrix-matrix addition](#matrix-matrix-addition)
    - [Matrix-scalar multiplication](#matrix-scalar-multiplication)
    - [Matrix-vector multiplication: dot product](#matrix-vector-multiplication-dot-product)
    - [Matrix-matrix multiplication](#matrix-matrix-multiplication)
    - [Matrix identity](#matrix-identity)
    - [Matrix inverse](#matrix-inverse)
    - [Matrix transpose](#matrix-transpose)
    - [Hadamard product](#hadamard-product)
- [Special matrices](#special-matrices)
    - [Rectangular matrix](#rectangular-matrix)
    - [Square matrix](#square-matrix)
    - [Diagonal matrix](#diagonal-matrix)
    - [Upper triangular matrix](#upper-triangular-matrix)
    - [Lower triangular matrix](#lower-triangular-matrix)
    - [Symmetric matrix](#symmetric-matrix)
    - [Identity matrix](#identity-matrix)
    - [Scalar matrix](#scalar-matrix)
    - [Null or zero matrix](#null-or-zero-matrix)
    - [Echelon matrix](#echelon-matrix)
    - [Antidiagonal matrix](#antidiagonal-matrix)
    - [Design matrix](#design-matrix)
- [Matrices as systems of linear equations](#matrices-as-systems-of-linear-equations)
- [The four fundamental matrix subsapces](#the-four-fundamental-matrix-subsapces)
    - [The column space](#the-column-space)
    - [The row space](#the-row-space)
    - [The null space](#the-null-space)
    - [The null space of the transpose](#the-null-space-of-the-transpose)
- [Solving systems of linear equations with matrices](#solving-systems-of-linear-equations-with-matrices)
    - [Gaussian Elimination](#gaussian-elimination)
    - [Gauss-Jordan Elimination](#gauss-jordan-elimination)
- [Matrix basis and rank](#matrix-basis-and-rank)
- [Matrix norm](#matrix-norm)

**[Linear and affine mappings](#linear-and-affine-mappings)**:

- [Linear mappings](#linear-mappings)
- [Examples of linear mappings](#examples-of-linear-mappings)
    - [Negation matrix](#negation-matrix)
    - [Reversal matrix](#reversal-matrix)
- [Examples of nonlinear mappings](#examples-of-nonlinear-mappings)
    - [Norms](#norms)
    - [Translation](#translation)
- [Affine mappings](#affine-mappings)
    - [Affine combination of vectors](#affine-combination-of-vectors)
    - [Affine span](#affine-span)
    - [Affine space and subspace](#affine-space-and-subspace)
    - [Affine mappings using the augmented matrix](#affine-mappings-using-the-augmented-matrix)
- [Special linear mappings](#special-linear-mappings)
    - [Scaling](#scaling)
    - [Reflection](#reflection)
    - [Shear](#shear)
    - [Rotation](#rotation)
- [Projections](#projections)
    - [Projections onto lines](#projections-onto-lines)
    - [Projections onto general subspaces](#projections-onto-general-subspaces)
    - [Projections as approximate solutions to systems of linear equations](#projections-as-approximate-solutions-to-systems-of-linear-equations)

**[Matrix decompositions](#matrix-decompositions)**:
- [LU decomposition](#lu-decomposition)
    - [Elementary matrices](#elementary-matrices)
    - [The inverse of elementary matrices](#the-inverse-of-elementary-matrices)
    - [LU decomposition as Gaussian Elimination](#lu-decomposition-as-gaussian-elimination)
    - [LU decomposition with pivoting](#lu-decomposition-with-pivoting)
- [QR decomposition](#qr-decomposition)
    - [Orthonormal basis](#orthonormal-basis)
    - [Orthonormal basis transpose](#orthonormal-basis-transpose)
    - [Gram-Schmidt Orthogonalization ](#gram-schmidt-orthogonalization)
    - [QR decomposition as Gram-Schmidt Orthogonalization](#qr-decomposition-as-gram-schmidt-orthogonalization)
- [Determinant](#determinant)
    - [Determinant as measures of volume](#determinant-as-measures-of-volume)
    - [The 2X2 determinant](#the-2-x-2-determinant)
    - [The NXN determinant](#the-n-x-n-determinant)
    - [Determinants as scaling factors](#determinants-as-scaling-factors)
    - [The importance of determinants](#the-importance-of-determinants)
- [Eigenthings](#eigenthings)
    - [Change of basis](#change-of-basis)
    - [Eigenvectors, Eigenvalues, and Eigenspaces](#eigenvectors-eigenvalues-and-eigenspaces)
    - [Trace and determinant with eigenvalues](#trace-and-determinant-with-eigenvalues)
    - [Eigendecomposition](#eigendecomposition)
    - [Eigenbasis are a good basis](#eigenbasis-are-a-good-basis)
    - [Geometric interpretation of Eigendecomposition](#geometric-interpretation-of-eigendecomposition)
    - [The problem with Eigendecomposition](#the-problem-with-eigendecomposition)
- [Singular Value Decomposition](#singular-value-decomposition):
    - [Singular Value Decomposition Theorem](#singular-value-decomposition-theorem)
    - [Singular Value Decomposition computation](#singular-value-decomposition-computation)
    - [Geometric interpretation of the Singular Value Decomposition](#geometric-interpretation-of-the-singular-value-decomposition)
    - [Singular Value Decomposition vs Eigendecomposition](#singular-value-decomposition-vs-eigendecomposition)
- [Matrix Approximation](#matrix-approximation):
    - [Best rank-k approximation with SVD](#best-rank-k-approximation-with-svd)
    - [Best low-rank approximation as a minimization problem](#best-low-rank-approximation-as-a-minimization-problem)
    
**[Epilogue](#epilogue)**

# Preliminary concepts

While writing about linear mappings, I realized the importance of having a basic understanding of a few concepts before approaching the study of linear algebra. If you are like me, you may not have formal mathematical training beyond high school. If so, I encourage you to read this section and spent some time wrapping your head around these concepts before going over the linear algebra content (otherwise, you might prefer to skip this part). I believe that reviewing these concepts is of great help to understand the *notation*, which in my experience is one of the main barriers to understand mathematics for nonmathematicians: we are *non*native speakers, so we are continuously building up our vocabulary. I'll keep this section very short, as is not the focus of this mini-course.

For this section, my notes are based on readings of:

- **Geometric transformations (Vol. 1)** (1966) by Modenov & Parkhomenko
- **Naive Set Theory** (1960) by P.R. Halmos
- **Abstract Algebra: Theory and Applications** (2016) by Judson & Beeer. [Book link](http://abstract.pugetsound.edu/download/aata-20160809.pdf)

## Sets

Sets are one of the most fundamental concepts in mathematics. They are so fundamental that they are not defined in terms of anything else. On the contrary, other branches of mathematics are defined in terms of sets, including linear algebra. Put simply, **sets are well-defined collections of objects**. Such objects are called **elements or members** of the set. The crew of a ship, a caravan of camels, and the LA Lakers roster, are all examples of sets. The captain of the ship, the first camel in the caravan, and LeBron James are all examples of "members" or "elements" of their corresponding sets. We denote a set with an upper case italic letter as $\textit{A}$. In the context of linear algebra, we say that a line is a set of points, and the set of all lines in the plane is a set of sets. Similarly, we can say that *vectors* are sets of points, and *matrices* sets of vectors.

## Belonging and inclusion

We build sets using the notion of **belonging**. We denote that $a$ *belongs* (or is an *element* or *member* of) to $\textit{A}$ with the Greek letter epsilon as:

$$
a \in \textit{A}
$$

Another important idea is **inclusion**, which allow us to build *subsets*. Consider sets $\textit{A}$ and $\textit{B}$. When every element of $\textit{A}$ is an element of $\textit{B}$, we say that $\textit{A}$ is a *subset* of $\textit{B}$, or that $\textit{B}$ *includes* $\textit{A}$. The notation is:

$$
\textit{A} \subset \textit{B}
$$

or

$$
\textit{B} \supset \textit{A}
$$

Belonging and inclusion are derived from **axion of extension**: *two sets are equal if and only if they have the same elements*. This axiom may sound trivially obvious but is necessary to make belonging and inclusion rigorous.

## Set specification

In general, anything we assert about the elements of a set results in **generating a subset**. In other words, asserting things about sets is a way to manufacture subsets. Take as an example the set of all dogs, that I'll denote as $\textit{D}$. I can assert now "$d$ is black". Such an assertion is true for some members of the set of all dogs and false for others. Hence, such a sentence, evaluated for *all* member of $\textit{D}$, generates a subset: *the set of all black dogs*. This is denoted as:

$$
\textit{B} = \{ d \in \textit{D} : \text{d is black} \}
$$

or 

$$
\textit{B} = \{ d \in \textit{D} \vert \text{ d is black} \}
$$

The colon ($:$) or vertical bar ($\vert$) read as "such that". Therefore, we can read the above expression as: *all elements of $d$ in $\textit{D}$ such that $d$ is black*. And that's how we obtain the set $\textit{B}$ from $\textit{A}$. 

Set generation, as defined before, depends on the **axiom of specification**: *to every set $\textit{A}$ and to every condition $\textit{S}(x)$ there corresponds a set $\textit{B}$ whose elements are exactly those elements $a \in \textit{A}$ for which $\textit{S}(x)$ holds.*

A condition $\textit{S}(x)$ is any *sentence* or *assertion* about elements of $\textit{A}$. Valid sentences are either of *belonging* or *equality*. When we combine belonging and equality assertions with logic operators (not, if, and or, etc), we can build any legal set.  

## Ordered pairs 

Pairs of sets come in two flavors: *unordered* and *ordered*. We care about pairs of sets as we need them to define a notion of relations and functions (from here I'll denote sets with lower-case for convenience, but keep in mind we're still talking about sets).

Consider a pair of sets $\textit{x}$ and $\textit{y}$. An **unordered pair** is a set whose elements are $\{ \textit{x},\textit{y} \}$, and $\{ \textit{x},\textit{y} \} = \{ \textit{y},\textit{x} \} $. Therefore, presentation order does not matter, the set is the same.

In machine learning, we usually do care about presentation order. For this, we need to define an **ordered pair** (I'll introduce this at an intuitive level, to avoid to introduce too many new concepts). An **ordered pair** is denoted as $( \textit{x},\textit{y} )$, with $\textit{x}$ as the *first coordinate* and $\textit{y}$ as the *second coordinate*. A valid ordered pair has the property that $( \textit{x},\textit{y} ) \ne ( \textit{y},\textit{x} )$.

## Relations

From ordered pairs, we can derive the idea of **relations** among sets or between elements and sets. Relations can be binary, ternary, quaternary, or N-ary. Here we are just concerned with binary relationships. In set theory, **relations** are defined as *sets of ordered pairs*, and denoted as $\textit{R}$. Hence, we can express the relation between $\textit{x}$ and $\textit{y}$ as:

$$
\textit{x R y}
$$

Further, for any $\textit{z} \in \textit{R}$, there exist $\textit{x}$ and $\textit{y}$ such that $\textit{z} = (\textit{x}, \textit{y})$. 

From the definition of $\textit{R}$, we can obtain the notions of **domain** and **range**. The **domain** is a set defined as:

$$
\text{dom } \textit{R} = \{ \textit{x:  for some y } ( \textit{x R y)} \}
$$

This reads as: the values of $\textit{x}$ such that for at least one element of $\textit{y}$, $\textit{x}$ has a relation with $\textit{y}$. 

The **range** is a set defined as:

$$
\text{ran } \textit{R} = \{ \textit{y:  for some x } ( \textit{x R y)} \}
$$

This reads: the set formed by the values of $\text{y}$ such that at least one element of $\textit{x}$, $\textit{x}$ has a relation with $\textit{y}$. 

## Functions

Consider a pair of sets $\textit{X}$ and $\textit{Y}$. We say that a **function** from $\textit{X}$ to $\textit{Y}$ is relation such that:

- $dom \textit{ f} = \textit{X}$ and
- such that for each $\textit{x} \in \textit{X}$ there is a unique element of  $\textit{y} \in \textit{Y}$ with $(\textit{x}, \textit{y}) \in {f}$ 

More informally, we say that a function "*transform*" or "*maps*" or "*sends*" $\textit{x}$ onto $\textit{y}$, and for each "*argument*" $\textit{x}$ there is a unique value $\textit{y}$ that $\textit{f }$ "*assummes*" or "*takes*".

We typically denote a relation or function or transformation or mapping from X onto Y as:

$$
\textit{f}: \textit{X} \rightarrow \textit{Y}
$$

or

$$
\textit{f}(\textit{x}) = \textit{y} 
$$

The simples way to see the effect of this definition of a function is with a chart. In **Fig. 1**, the left-pane shows a valid function, i.e., each value $\textit{f}(\textit{x})$ *maps* uniquely onto one value of $\textit{y}$. The right-pane is not a function, since each value $\textit{f}(\textit{x})$ *maps* onto multiple values of $\textit{y}$. 

**Fig. 1: Functions**


<img src="/assets/post-10/b-function.svg">


For $\textit{f}: \textit{X} \rightarrow \textit{Y}$, the *domain* of $\textit{f}$ equals to $\textit{X}$, but the *range* does not necessarily equals to  $\textit{Y}$. Just recall that the *range* includes only the elements for which $\textit{Y}$ has a relation with $\textit{X}$. 

**The ultimate goal of machine learning is learning functions from data**, i.e., transformations or mappings from the *domain* onto the *range* of a function. This may sound simplistic, but it's true. The *domain* $\textit{X}$ is usually a vector (or set) of *variables* or *features* mapping onto a vector of *target* values. Finally, I want to emphasize that in machine learning the words transformation and mapping are used interchangeably, but both just mean function.

This is all I'll cover about sets and functions. My goals were just to introduce: (1) **the concept of a set**, (2) **basic set notation**, (3) **how sets are generated**, (4) **how sets allow the definition of functions**, (5) **the concept of a function**. Set theory is a monumental field, but there is no need to learn everything about sets to understand linear algebra. Halmo's **Naive set theory** (not free, but you can find a copy for ~\\$8-$10 US) is a fantastic book for people that just need to understand the most fundamental ideas in a relatively informal manner.  


```python
# Libraries for this section 
import numpy as np
import pandas as pd
import altair as alt
alt.themes.enable('dark')
```




    ThemeRegistry.enable('dark')



# Vectors

Linear algebra is the study of vectors. At the most general level, vectors are **ordered finite lists of numbers**. Vectors are the most fundamental mathematical object in machine learning. We use them to **represent attributes of entities**: age, sex, test scores, etc. We represent vectors by a bold lower-case letter like $\bf{v}$ or as a lower-case letter with an arrow on top like $\vec{v}$.

Vectors are a type of mathematical object that can be **added together** and/or **multiplied by a number** to obtain another object of **the same kind**. For instance, if we have a vector $\bf{x} = \text{age}$ and a second vector $\bf{y} = \text{weight}$, we can add them together and obtain a third vector $\bf{z} = x + y$. We can also multiply $2 \times \bf{x}$ to obtain $2\bf{x}$, again, a vector. This is what we mean by *the same kind*: the returning object is still a *vector*. 

## Types of vectors

Vectors come in three flavors: (1) **geometric vectors**, (2) **polynomials**, (3) and **elements of $\mathbb{R^n}$ space**. We will defined each one next.

### Geometric vectors

**Geometric vectors are oriented segments**. Therse are the kind of vectors you probably learned about in high-school physics and geometry. Many linear algebra concepts come from the geometric point of view of vectors: space, plane, distance, etc.

**Fig. 2: Geometric vectors**


<img src="/assets/post-10/b-geometric-vectors.svg">


### Polynomials

**A polynomial is an expression like $f(x) = x^2 + y + 1$**. This is, a expression adding multiple "terms" (nomials). Polynomials are vectors because they meet the definition of a vector: they can be added together to get another polynomial, and they can be multiplied together to get another polynomial. 

$$
\text{function addition is valid} \\
f(x) + g(x)\\
$$
$$
and\\
$$
$$
\text{multiplying by a scalar is valid} \\
5 \times f(x)
$$

**Fig. 3: Polynomials**


<img src="/assets/post-10/b-polynomials-vectors.svg">


### Elements of R

**Elements of $\mathbb{R}^n$ are sets of real numbers**. This type of representation is arguably the most important for applied machine learning. It is how data is commonly represented in computers to build machine learning models. For instance, a vector in $\mathbb{R}^3$ takes the shape of:

$$
\bf{x}=
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
\in \mathbb{R}^3
$$

Indicating that it contains three dimensions.

$$
\text{addition is valid} \\
\phantom{space}\\
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} +
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}=
\begin{bmatrix}
2 \\
4 \\
6
\end{bmatrix}\\
$$
$$
and\\
$$
$$
\text{multiplying by a scalar is valid} \\
\phantom{space}\\
5 \times
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix} = 
\begin{bmatrix}
5 \\
10 \\
15
\end{bmatrix}
$$

In `NumPy` vectors are represented as n-dimensional arrays. To create a vector in $\mathbb{R^3}$:


```python
x = np.array([[1],
              [2],
              [3]])
```

We can inspect the vector shape by:


```python
x.shape # (3 dimensions, 1 element on each)
```




    (3, 1)




```python
print(f'A 3-dimensional vector:\n{x}')
```

    A 3-dimensional vector:
    [[1]
     [2]
     [3]]


## Zero vector, unit vector, and sparse vector

There are a couple of  "special" vectors worth to remember as they will be mentioned frequently on applied linear algebra: (1) zero vector, (2) unit vector, (3) sparse vectors

**Zero vectors**, are vectors composed of zeros, and zeros only. It is common to see this vector denoted as simply $0$, regardless of the dimensionality. Hence, you may see a 3-dimensional or 10-dimensional with all entries equal to 0, refered as "the 0" vector. For instance:

$$
\bf{0} = 
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
$$

**Unit vectors**, are vectors composed of a single element equal to one, and the rest to zero. Unit vectors are important to understand applications like norms. For instance, $\bf{x_1}$, $\bf{x_2}$, and $\bf{x_3}$ are unit vectors:

$$
\bf{x_1} = 
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix},
\bf{x_2} = 
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix},
\bf{x_3} = 
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}
$$

**Sparse vectors**, are vectors with most of its elements equal to zero. We denote the number of nonzero elements of a vector $\bf{x}$ as $nnz(x)$. The sparser possible vector is the zero vector. Sparse vectors are common in machine learning applications and often require some type of method to deal with them effectively.  


## Vector dimensions and coordinate system

Vectors can have any number of dimensions. The most common are the 2-dimensional cartesian plane, and the 3-dimensional space. Vectors in 2 and 3 dimensions are used often for pedgagogical purposes since we can visualize them as geometric vectors. Nevetheless, most problems in machine learning entail more dimensions, sometiome hundreds or thousands of dimensions. The notation for a vector $\bf{x}$ of arbitrary dimensions, $n$ is:

$$
\bf{x} = 
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
\in \mathbb{R}^n
$$

Vectors dimensions map into **coordinate systems or perpendicular axes**. Coordinate systems have an origin at $(0,0,0)$, hence, when we define a vector:

$$\bf{x} = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix} \in \mathbb{R}^3$$

we are saying: starting from the origin, move 3 units in the 1st perpendicular axis, 2 units in the 2nd perpendicular axis, and 1 unit in the 3rd perpendicular axis. We will see later that when we have a set of perpendicular axes we obtain the basis of a vector space.

**Fig. 4: Coordinate systems**


<img src="/assets/post-10/b-coordinate-system.svg">


## Basic vector operations

### Vector-vector addition 

We used vector-vector addition to define vectors without defining vector-vector addition. Vector-vector addition is an element-wise operation, only defined for vectors of the same size (i.e., number of elements). Consider two vectors of the same size, then: 

$$
\bf{x} + \bf{y} = 
\begin{bmatrix}
x_1\\
\vdots\\
x_n
\end{bmatrix}+
\begin{bmatrix}
y_1\\
\vdots\\
y_n
\end{bmatrix} =
\begin{bmatrix}
x_1 + y_1\\
\vdots\\
x_n + y_n
\end{bmatrix}
$$

For instance:

$$
\bf{x} + \bf{y} = 
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}+
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix} =
\begin{bmatrix}
1 + 1\\
2 + 2\\
3 + 3
\end{bmatrix} =
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

Vector addition has a series of **fundamental properties** worth mentioning:

1. Commutativity: $x + y = y + x$
2. Associativity: $(x + y) + z = x + (y + z)$
3. Adding the zero vector has no effect: $x + 0 = 0 + x = x$
4. Substracting a vector from itself returns the zero vector: $x - x = 0$

In `NumPy`, we add two vectors of the same with the `+` operator or the `add` method:


```python
x = y = np.array([[1],
                  [2],
                  [3]])
```


```python
x + y
```




    array([[2],
           [4],
           [6]])




```python
np.add(x,y)
```




    array([[2],
           [4],
           [6]])



### Vector-scalar multiplication
