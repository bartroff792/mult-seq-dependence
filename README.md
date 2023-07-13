# Sequential Testing of Multiple Hypotheses
Extremely broken at the moment. We're working on it!

# Structure
The only parts that are close to functional are in the Code/utils directory.
They inlcude

* `cutoff_funcs`: functions for building vectors of p-value cutoffs and log likelihood ratio cutoffs for sequential testing procedures
* `data_funcs`: Funcs for reading drug data, generating fake data, generating hypotheses, and computing llr paths.
* `simulation_orchestration`: This module contains functions for higher level simulation for sequential testing of multiple hypotheses (beyond just generating the observations), and executing the SPRT procedures on it.
* `sim_analysis`: consider this broken... still trying to remember what this does. most functions are basically undocumented
* `common_funcs`: not much other that a function for chunking lists of jobs

Outside of that, the smattering of .pys in the main Code dir mostly run sets of simulations, generate plots and dump them, though they're all a mess. The only other module to look at at the moment is the `visualizations.py` module.

Under `Data`, we have 

* `AmnesiaRateClean.csv`: a table of drugs, each with the (annual) rate at which they've "generated" amnesia side effect reports, as well as the rate at which they've generated non-amnesia side effect reports. We recommend using their total side effect generation rate as a proxy for their usage.
* `GoogleSearchHitData.csv`: a table of drugs search popularity, and the proportion of those searches that include "amnesia". The popularity and naming schemes of drugs differ, so some of these may be of higher than expected variance. Further, many drugs' search rates weren't available.
* `YellowcardData.csv`: a bit too raw... contains number of total side effects, fatal side effects, amnesia reports, etc for each drug.

# BL scaling


Taking 
* $m_{0}$ to be the number of true null hypotheses
* $m_{1}$ to be the number of false null hypotheses
* $\vec{\alpha}=(\alpha_{1}, \alpha_{2}, ... \alpha_{m_{0}+ m_{1}})$ to be the vector of p-value cutoffs such that $\alpha_{j}\leq \alpha_{j+1}$

Then define the Guo+Rao FDR bound for a stepdown procedure to be  
$$
D(m_{0},m_{1},\vec{\alpha})=m_{0}\left(\sum_{j=1}^{m_{1}+1}\frac{\alpha_{j}-\alpha_{j-1}}{j}+\sum_{j=m_{1}+2}^{m}\frac{m_{1}(\alpha_{j}-\alpha_{j-1})}{j(j-1)}\right)
$$
when $m_{0}$ (and $m_{1}$) are known, and 
$$
D(\vec{\alpha}) = \max_{m_{0}\in \left\{1,...,m\right\}} D\left(m_{0}, m - m_{0}, \vec{\alpha}\right)
$$
when they're unknown.

# Main Theorems

## THM Finite Horizon Rejective $fdr$ Control

The finite horizon, rejective sequential step-down procedure described
in Algorithm \ref{alg:Finite-Horizon-Rejective} with $\alpha$ cutoffs
as in Equation \ref{eq:alpha-cutoffs} and cutoffs $A_{j}$ as specified
in Equation \ref{eq:rejective-cutoffs} satisfying 

$$
\forall i\leq m,\theta\in H_{0}^{i}\quad P_{\theta}(\exists t<T\text{ s.t. }\Lambda^{i}(t)\geq A_{j})\leq\alpha_{j}
$$
provides the type 1 error bound 
$$
fdr\leq D\left(\vec{\alpha}\right)
$$
under the true distribution, regardless of the dependence structure,
where $D\left(\vec{\alpha}\right)$ is defined as in Equation \ref{eq:fdr-d-func}.

## THM Infinite Horizon $fdr$ and $fnr$ Control

The infinite horizon variant is similar, but requires slightly different
marginal inequalities and provides type 2 error control as well. 


The infinite horizon, acceptive-rejective sequential step-down procedure
described in Algorithm \ref{alg:Full-SSD} with $\vec{\alpha}$ and
$\vec{\beta}$ satisfying Equation \ref{eq:alpha-cutoffs} (and an
equivalent for type 2 error) and $A_{j}$'s and $B_{j}$'s as in Equations
\ref{eq:rejective-cutoffs} and \ref{eq:acceptance-cutoffs} satisfying 

$$
\forall i\leq m,\,j\leq m,\theta\in H_{0}^{i}\quad P_{\theta}(\exists t<\infty\text{ st }\Lambda^{i}(t)\geq A_{j}\,\cap\,\forall t^{\prime}<t\quad\Lambda(t^{\prime})>B_{1})\leq\alpha_{j}
$$

$$
\forall i\leq m,\,j\leq m,\theta\in H_{1}^{i}\quad P_{\theta}(\exists t<\infty\text{ st }\Lambda^{i}(t)\leq B_{j}\,\cap\,\forall t^{\prime}<t\quad\Lambda^{i}(t^{\prime})<A_{1})\leq\beta_{j}
$$

provides the following bounds on type 1 and type 2 error under the
true distribution, regardless of the dependence structure:

$$
fdr\leq D(\vec{\alpha})
$$

$$
fnr\leq D(\vec{\beta}).
$$

Again, $D\left(\vec{\alpha}\right)$ is defined as in Equation \ref{eq:fdr-d-func},
and $D\left(\vec{\beta}\right)$ is the same function applied to the
type 2 marginal constraints vector.
\end{thm}
These theorems allow for construction of $fdr$ and $fnr$ controlling
procedures via the scaling approach described in Equation \ref{eq:new-alpha-vec}.
In the case of an SPRT, the $\alpha$ cutoffs can then be used to
construct the statistic cutoffs using the multiple hypothesis Wald
approximations discussed in Section \ref{subsec:Bartroff-Multiple-Wald}.

We also present similar theorems for $pfdr$ and $pfnr$ control,
and corollaries for application of it to SPRT and more general tests.
## THM: Finite Horizon Rejective $pfdr$ Control

The finite horizon, rejective sequential step-down procedure described
in Algorithm \ref{alg:Finite-Horizon-Rejective} with $\alpha$ cutoffs
as in Equation \ref{eq:alpha-cutoffs} and test statistic cutoffs
$A_{j}$ as specified in Equation \ref{eq:rejective-cutoffs} satisfying
the marginal constraints in Equation \ref{eq:FDR finite horizon constraint}
provides the type 1 error bound under the true distribution with $m_{0}$
true null hypotheses and $m_{1}$ false null hypotheses, regardless
of the dependence structure:

$$
pfdr\leq D\left(m_{0},m_{1},\vec{\alpha}/P\left(R>0\right)\right)=D\left(m_{0},m_{1},\vec{\alpha}\right)/P\left(R>0\right).
$$

Further, we also have

$$
pfdr\leq\frac{D\left(m_{0},m_{1},\vec{\alpha}\right)}{\max_{1\leq i\leq m}P\left(\exists t<T\,\text{ st }\,\Lambda^{i}(t)\geq A_{1}\right)}\leq\frac{D\left(m_{0},m_{1},\vec{\alpha}\right)}{\min_{1\leq i\leq m}P\left(\exists t<T\,\text{ st }\,\Lambda^{i}(t)\geq A_{1}\right)}.
$$

## THM: $pfdr$ and $pfnr$ Control for Infinite Horizon

In the finite horizon, rejective procedure, the quantity $P\left(R>0\right)$
is equivalent to 

$$
P\left(\exists i\in 1,...,m,\,t<T\,\text{ st }\,\Lambda^{i}\left(t\right)\geq A_{1}\right),
$$

and depends on the procedure and cutoffs. The implications of this
theorem on determining the actual level of $pfdr$ control are discussed
in Section \ref{subsec:pfdr-Control-Estimation}.

The infinite horizon, acceptive-rejective sequential step-down procedure
described in Algorithm \ref{alg:Full-SSD}, with $\vec{\alpha}$ and
$\vec{\beta}$ satisfying Equation \ref{eq:alpha-cutoffs} (and an
equivalent for type 2 error) and $A_{j}$'s and $B_{j}$'s as in Equations
\ref{eq:rejective-cutoffs} and \ref{eq:acceptance-cutoffs} satisfying
the marginal constraints in Equations \ref{eq:FDR Infinite Horizon Type 1 Constraint}
and \ref{eq:FNR Infinite Horizon Type 2 Constraint}, provides the
type 1 and type 2 error bounds under the true distribution with $m_{0}$
true null hypotheses and $m_{1}$ false null hypotheses, regardless
of the dependence structure:

$$\begin{aligned}
pfdr & \leq D\left(m_{0},m_{1},\frac{\vec{\alpha}}{P\left(R>0\right)}\right)\\
 & \leq\frac{D\left(m_{0},m_{1},\vec{\alpha}\right)}{\max_{1\leq i\leq m}P\left(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})>B_{m}\right)}\\
 & \leq\frac{D\left(m_{0},m_{1},\vec{\alpha}\right)}{P\left(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})>B_{m}\right)}\quad\forall i\in 1,...,m,
\end{aligned}$$

$$\begin{aligned}
pfnr & \leq D\left(m_{1},m_{0},\frac{\vec{\beta}}{P\left(R^{\prime}>0\right)}\right)\\
 & \leq\frac{D\left(m_{1},m_{0},\vec{\beta}\right)}{\max_{1\leq i\leq m}P\left(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\leq B_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})<A_{m}\right)}\\
 & \leq\frac{D\left(m_{1},m_{0},\vec{\beta}\right)}{P\left(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\leq B_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})<A_{m}\right)}\quad\forall i\in 1,...,m.
\end{aligned}$$

Note that in the case of a simple vs. simple SPRT with both $m_{1}>0$
and $m_{0}>0$, we may apply Wald's approximation to the denominator
of the bounds above

$$
P_{H_{0}^{i}}(\exists t<\infty\text{ st }\Lambda^{i}(t)\leq B_{1},\forall t^{\prime}<t\quad\Lambda^{i}(t^{\prime})<A_{m})\\
=1-P_{H_{0}^{i}}(\left(\forall t<\infty\quad\Lambda^{i}(t)>B_{1}\right)\,\cup\,\left(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{m},\,\forall t^{\prime}<t\quad\Lambda^{i}(t^{\prime})>B_{1}\right))\\
=1-P_{H_{0}^{i}}(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{m},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})>B_{1})\approx1-\alpha_{m}.
$$


Using this approximate equivalence and we may then achieve the following
approximate bounds, subject to the conditions in Theorem \ref{thm:pfdr-infinite}:

$$
pfdr\lesssim\frac{D(m_{0},m_{1},\vec{\alpha})}{1-\beta_{m}}
$$

$$
pfnr\lesssim\frac{D(m_{1},m_{0},\vec{\beta})}{1-\alpha_{m}},
$$
where $\lesssim$ is understood to mean ``less than or on the order
of.''

For arbitrary test statistics, we may add power conditions requiring
the existence of a lower bound, the probability that a false (true)
null is correctly rejected (accepted) at the most extreme level, and
thus obtain the following more general bounds:

## THM $pfdr$ and $pfnr$ Control for Infinite Horizon with General
Test Statistics

In addition to the assumptions presented in Theorem \ref{thm:pfdr-infinite},
given the following lower-bounds on accurate rejection and acceptance
$\forall i\leq m$:

$$
\exists i\in 1,...,m,\,\text{ st }\,\forall\theta\in H_{1}^{i}\quad P_{\theta}(\exists t<\infty\text{ st }\Lambda^{i}(t)\geq A_{1},\,\Lambda^{i}(t^{\prime})>B_{m}\quad\forall t^{\prime}<t)\geq\gamma_{1}
$$


$$
\exists i\in 1,...,m,\,\text{ st }\,\forall\theta\in H_{0}^{i}\quad P_{\theta}(\exists t<\infty\text{ st }\Lambda^{i}(t)\leq B_{1},\,\Lambda^{i}(t^{\prime})<A_{m}\quad\forall t^{\prime}<t)\geq\gamma_{0}
$$
provides the type 1 and type 2 error bounds under the true distribution
with $m_{0}$ true null hypotheses and $m_{1}$ false null hypotheses,
regardless of the dependence structure

$$
pfdr\leq\frac{D(m_{0},m_{1},\vec{\alpha})}{\gamma_{1}}
$$

$$
pfnr\leq\frac{D(m_{1},m_{0},\vec{\beta})}{\gamma_{0}}.
$$
