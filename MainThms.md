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
fdr\leq D(\vec{\alpha})
$$
under the true distribution, regardless of the dependence structure,
where $D(\vec{\alpha})$ is defined as in Equation \ref{eq:fdr-d-func}.

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

Again, $D(\vec{\alpha})$ is defined as in Equation \ref{eq:fdr-d-func},
and $D(\vec{\beta})$ is the same function applied to the
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
pfdr\leq D(m_{0},m_{1},\vec{\alpha}/P(R>0))=D(m_{0},m_{1},\vec{\alpha})/P(R>0).
$$

Further, we also have

$$
pfdr 
\leq
\frac{
    D(m_{0},m_{1},\vec{\alpha})
    }{
    \max_{1\leq i\leq m}
        P(\exists t<T\,\text{ st }\,\Lambda^{i}(t)\geq A_{1})
    }
\leq
\frac{
    D(m_{0},m_{1},\vec{\alpha})
    }{
    \min_{1\leq i\leq m}
        P(\exists t<T\,\text{ st }\,\Lambda^{i}(t)\geq A_{1})
    }.
$$

## THM: $pfdr$ and $pfnr$ Control for Infinite Horizon

In the finite horizon, rejective procedure, the quantity $P(R>0)$
is equivalent to

$$
P(\exists i\in 1,...,m,\,t<T\,\text{ st }\,\Lambda^{i}(t)\geq A_{1}),
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
pfdr & \leq D(m_{0},m_{1},\frac{\vec{\alpha}}{P(R>0)})\\
 & \leq\frac{D(m_{0},m_{1},\vec{\alpha})}{\max_{1\leq i\leq m}P(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})>B_{m})}\\
 & \leq\frac{D(m_{0},m_{1},\vec{\alpha})}{P(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})>B_{m})}\quad\forall i\in 1,...,m,
\end{aligned}$$

$$\begin{aligned}
pfnr & \leq D(m_{1},m_{0},\frac{\vec{\beta}}{P(R^{\prime}>0)})\\
 & \leq\frac{D(m_{1},m_{0},\vec{\beta})}{\max_{1\leq i\leq m}P(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\leq B_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})<A_{m})}\\
 & \leq\frac{D(m_{1},m_{0},\vec{\beta})}{P(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\leq B_{1},\,\forall t^{\prime}<t\,\Lambda^{i}(t^{\prime})<A_{m})}\quad\forall i\in 1,...,m.
\end{aligned}$$

Note that in the case of a simple vs. simple SPRT with both $m_{1}>0$
and $m_{0}>0$, we may apply Wald's approximation to the denominator
of the bounds above

$$
P_{H_{0}^{i}}(\exists t<\infty\text{ st }\Lambda^{i}(t)\leq B_{1},\forall t^{\prime}<t\quad\Lambda^{i}(t^{\prime})<A_{m})\\
=1-P_{H_{0}^{i}}((\forall t<\infty\quad\Lambda^{i}(t)>B_{1})\,\cup\,(\exists t<\infty\,\text{ st }\,\Lambda^{i}(t)\geq A_{m},\,\forall t^{\prime}<t\quad\Lambda^{i}(t^{\prime})>B_{1}))\\
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
