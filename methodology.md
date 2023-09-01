# Handling Negative Noise with NMF
----

## Algorithm 1

Non-negative matrix factorization (NMF) requires that all data be non-negative, which is fundamentally incompatible with postprocessed spectroscopic data that can legitimately include some negative values. In general, true emission and continuum features will be non-negative, but noise or sky subtraction can drive some pixels into a negative regime. When fitting a clipped version of this data, where negative values are set to 0, we will end up with inaccurate templates that believe the noise mean is some value $\geq 0$.

Assuming that the noise is zero-mean, we can account for this noise using a \"shift and deshift\" methodology, which allows template fitting to consider the normally negative values when fitting templates while still restricting those templates to lie in the upper half plane. While this was developed in the context of spectroscopic data, it can be generalized to apply to any data with some negative component.

The methodology is as follows:

1.  Vertically shift the data so that all elements are non-negative. The shift value will depend on the input spectra, and should be equal to or more than the lowest negative value in the data matrix being fit.

2.  Fit the desired NMF templates with an additional fixed and constant template. The fixed template is a constant value equivalent to the shift, with a constant coefficient fixed at 1.

After the fitting procedure the fixed template can be discarded, and the NMF fit to the original (unshifted) data is then the found coefficients and templates, without the fixed template included.

For this note I will adopt a similar convention as Zhu 2016[^1]. All matrices will be represented by uppercase letters, and scalars by lowercase. We are training a matrix $W$ of $q$ basis vectors (alternatively called templates) and a matrix $H$ of $N$ coefficients, designed to approximate the $d \times N$ matrix of data $X$, where each spectra is a column of $X$. $d$ represents the dimensionality of the data, so that $H$ is $q\times N$ and $W$ is $d \times q$. $V$ is a matrix of pixel specific weights with the same shape as $X$. Shifting $X$ by a value $y_0$ moves us to the shifted frame, represented by $X^{\prime} = X + y_0$. In the shifted frame, the objective function to minimize is given by

$$ \chi^2 = ||V^{1/2} \circ (X^{\prime} - WH - W_0H_0) ||^2$$

where I have introduced $W_0$ as the aforementioned constant template with magnitude equivalent to the shift $y_0$, and $H_0$ as the $1\times N$ vector of ones corresponding to the template's fixed coefficient. $\circ$ represents elementwise multiplication, and the square root on $V$ is applied elementwise. The objective function given above is locally convex in either the $W$ or $H$ directions, so each can be solved for individually.

### H update

To derive the update rule we begin by taking the derivative of the objective with respect to the matrix $H$ and setting it equal to zero:
$$\frac{\partial \chi^2}{\partial H} = W^T (V \circ (X^{\prime} - WH - W_0H_0)) = 0,$$

where I have already dropped the numerical factor of $-2$. With some simple, but not exhaustive, algebraic manipulation we can find then that a single optimization step for $H$ is given by

$$\begin{aligned}
0 &= W^T (V \circ (X^{\prime} - WH - W_0H_0)) \\
W^T (V\circ X^{\prime}) &= W^T(V \circ (WH + {W_0H_0})) \\
1 &= \frac{W^T (V\circ X^{\prime})}{W^T(V \circ (WH + {W_0H_0}))}\end{aligned}$$

$$\boxed{H \leftarrow H \circ \frac{W^T (V\circ X^{\prime})}{W^T(V \circ (WH + {W_0H_0}))}}$$

With division applied elementwise. Step 3 holds as long as $H$ is truly a minimizer, with step 4 following from 3 as the step necessary to move $H$ towards that minima. Evidently when $H$ is a minimizer it remains unchanged. For more intuition and a more robust proof of this update rule, see Tang et al. 2012 [^2].

### W update

The $W$ update is derived using the same method as the $H$ update, so I will skip the algebra here.

$$\frac{\partial \chi^2}{\partial W} = (V \circ (X^{\prime} - WH - W_0H_0))H^T = 0$$

$$\boxed{W \leftarrow W \circ \frac{(V\circ X^{\prime})H^T}{(V \circ (WH + {W_0H_0}))H^T}}$$

These two update rules ($W$ and $H$) are independent, and can be used in tandem or individuallty. For fitting data to an already trained set of templates, it is necessary to only update the $H$ vector of coefficients, while holding the templates $W$ fixed without update. For training templates to fit to data, it is necessary to use **both** update rules. The usual methodology is to alternate, and use the updated version of one in the other until convergence, but it is not a requirement to train both in a 1:1 ratio.

For a proof of convergence of these udpate rules see the appendix of Tang et al. 2012[^2], which allows the coefficients of the fixed templates, $H_0$ in this note, to vary and which does not include weighting, but for which the proof still broadly applies to this case.

### Code

The following small Python code block represents the smallest reproducible example of the \"shift and deshift\" methodology presented in this note:

    # Initialize H and W before this code block
    X = X + shift
    H_0 = np.ones((1, X.shape[1]))
    W_0 = np.ones((X.shape[0], 1)) * shift

    # Weighted X, outside the loop for efficiency
    V_X = V * X
    V_WH_0 = V * (W_0 @ H_0)
    for _ in range(n_iter):
        H = H * (W.T @ (V_X)) / (W.T @ (V * (W @ H) + V_WH_0))
        W = W * ((V_X) @ H.T) / ((V * (W @ H) + V_WH_0) @ H.T)

Some subtleties have been excised from this code. For example, with some weights set to zero, a division by zero can cause a \"nan\" to propagate in the update rules. This should be carefully handled by setting those values back to 0 after the update.

The above represents a direct code translation of the conceptual method. For efficiency and optimization it is prudent to recognize that the fixed template/coefficient combination is equivalent to adding a scalar into the update rule, and so code can be reduced to:

    # Initialize H and W before this code block
    X = X + shift
    # Weighted X, outside the loop for efficiency
    V_X = V * X
    for _ in range(n_iter):
        H = H * (W.T @ (V_X)) / (W.T @ (V * (W @ H + shift)))
        W = W * ((V_X) @ H.T) / ((V * (W @ H + shift)) @ H.T)

When fitting a set of data with a fixed set of templates, we can excise the template update rule accordingly:

    # Initialize H and W before this code block
    X = X + shift
    # Weighted X, outside the loop for efficiency
    V_X = V * X
    for _ in range(n_iter):
        H = H * (W.T @ (V_X)) / (W.T @ (V * (W @ H + shift)))

Here `n_iter` can be reduced to some small amount ($\lesssim 5$) since convergence will be fast in this singular direction.

----

## Algorithm 2

In this example, we will have one set of "regular" NMF templates, and one set with nonnegative templates but with coefficients that may vary either positive *or* negative. Thus the objective function is $$\chi^2 = ||X - W_1H_1 - W_2H_2||^2$$

For simplicity of the first pass derivation I will exclude weights. $W_1$ and $H_1$ will be the non-negative templates and coefficients, $W_2$ will be the fixed nonnegative template(s) that are allowed to have negative coefficients, $H_2$. We will allow $X$ to be negative. To summarize:

$$X, H_2 \in \mathbb{R}_{\pm}; H_1, W_1, W_2 \in \mathbb{R}_+ $$

Following the methodology of Tang et al. 2012 for the $W_1$ and $H_1$ updates we can separate out the derivatives into positive and negative components to create the update rules. For $H_1$:

$$ \begin{aligned}
\frac{\partial \chi^2}{\partial H_1} &= -2 W_1^T(X - W_1H_1 - W_2H_2) \\
\left[\frac{\partial \chi^2}{\partial H_1}\right]^+ &= [W_1^TX]^- + [W_1^T W_2H_2]^+ + W_1^TW_1H_1\\
\left[\frac{\partial \chi^2}{\partial H_1}\right]^-& = [W_1^TX]^+ + [W_1^T W_2H_2]^-\end{aligned}$$

$[A]^+$ represents the positive elements of matrix $A$ (with all other elements set to 0), and likewise $[A]^-$ represents the negative elements of matrix $A$ (set to be positive, with all other elements set to 0). A simple mathematic way to represent this is given by (originally from Ding et al 2010)[^3]:

$$\begin{aligned}
[A]^+ &= (|A| + A) / 2 \\
[A]^- &= (|A| - A) / 2 \end{aligned}$$



So the update rule is given by (from Tang et al. 2012):
$$
H_1 \leftarrow H_1 \circ \frac{\left[\frac{\partial \chi^2}{\partial H_1}\right]^-}{\left[\frac{\partial \chi^2}{\partial H_1}\right]^+}$$

and we can plug in our found values to get
$$
H_1 \leftarrow H_1 \circ \frac{[W_1^TX]^+ + [W_1^T W_2H_2]^-}{[W_1^TX]^- + [W_1^T W_2H_2]^+ + W_1^TW_1H_1}
$$

Note that, as expected, when there is no negative data in $X$, and we do not include the additional templates, this reduces directly to the multiplicative update rule for NMF derived by Lee and Seung. Similar algebra produces the update rule for $W_1$:

$$W_1 \leftarrow W_1 \circ \frac{[XH_1^T]^+ + [W_2H_2H_1^T]^-}{[XH_1^T]^- + [W_2H_2H_1^T]^+ + W_1H_1H_1^T}$$

which again reduces to the familiar form under the correct constraints. These forms also propose an alternate method by which to deal with negative data. If we remove the additional basis vectors that we allow to have negative coefficients, we get update rules for $W_1$ and $H_1$ that allow $X$ to vary both positve and negative:

$$ \begin{aligned}
H_1 &\leftarrow H_1 \circ \frac{[W_1^TX]^+}{[W_1^TX]^- + W_1^TW_1H_1} \\
W_1 &\leftarrow W_1 \circ \frac{[XH_1^T]^+}{[XH_1^T]^-  + W_1H_1H_1^T} \end{aligned}
$$

A final step in our update rules is to derive the update rule for $H_2$. Since we allow $H_2$ to be unconstrained, this is equivalent to solving a regular least squares optimization problem, for which the minimizing solution to $H_2$ given values of $H_1$ and $W_1$ is

$$H_2 = (W_2^T W_2)^{-1} W_2^T(X - W_1H_1)$$

Note that in the very rare case that $(W_2^T W_2)$ is singular we should use the psuedoinverse instead. A proof that this update rule minimizes $\chi^2$ in a nonincreasing manner is included in Ding et al 2010.

Adding weights to the update rules to bring everything together we get update rules for nonnegative coefficients and templates, on possibly negative data, with a fixed positive template that may have negative coefficients:

$$\boxed{H_1 \leftarrow H_1 \circ \frac{[W_1^T (V\circ X)]^+ + [W_1^T (V\circ (W_2H_2))]^-}{[W_1^T(V\circ X)]^- + [W_1^T (V\circ (W_2H_2))]^+ + W_1^T (V\circ (W_1H_1))} }$$

$$\boxed{W_1 \leftarrow W_1 \circ \frac{[(V\circ X)H_1^T]^+ + [(V\circ (W_2H_2))H_1^T]^-}{[(V\circ X)H_1^T]^- + [(V\circ (W_2H_2))H_1^T]^+ + (V \circ (W_1H_1))H_1^T}}$$

$$\boxed{H_{2,i} = (W_2^T [V_i]^{diag} W_2)^{-1}W_2^T (V_i \circ \Phi_i) }$$

Where in the $H_2$ update rule we update each exposures coefficients individually to account for the weights varying per exposure. I have introduced $\Phi = X - (W_1H_1)$ to simplify notation. $[V_i]^{diag}$ is a diagonal matrix with the elements of $V_i$ along the diagonal.

In matrix notation, with some loss of understanding, this can be expressed as:

$$\hat{H}_2 = (\hat{W}_2^T [V]^{diag} \hat{W}_2)^{-1} \hat{W}_2^T [V]^{diag}\hat{\Phi}$$

$\hat{W}_2$ is a block diagonal, but not square, matrix constructed with the $W_2$ matrix $N$ times along the diagonal, and the rest of the elements set to 0, with a final dimensionality of $dN \times qN$. $[V]^{diag}$ is the $dN \times dN$ diagonal matrix constructed by concatenating every exposure's pixelwise weights and placing the elements along the diagonal, and $\hat{\Phi}$ is a vector formed by concatenating every column (exposure) of $\Phi = X - (W_1H_1)$. The resulting vector $\hat{H}_2$ is the vector resulting from concatenating each column of $H_2$, and returning it to matrix form is simply a matter of unfolding the concatenation back into a 2-dimensional matrix.

where $i$ indexes over each exposure. In essence, each exposure's $H_2$ coefficients depend on the weights for that exposure and the exposure minus its reconstruction from the strictly positive templates.


And the simplified update rules, for non-negative coefficients and templates on data that is allowed to be negative, with weights:

$$\boxed{H_1 \leftarrow H_1 \circ \frac{[W_1^T (V\circ X)]^+ }{[W_1^T(V\circ X)]^- + W_1^T (V\circ (W_1H_1))}}$$

$$\boxed{W_1 \leftarrow W_1 \circ \frac{[(V\circ X)H_1^T]^+ }{[(V\circ X)H_1^T]^- + (V \circ (W_1H_1))H_1^T}}$$


[^1]: Zhu, Guangtun. “Nonnegative Matrix Factorization (NMF) with Heteroscedastic Uncertainties and Missing Data.” arXiv, December 18, 2016. http://arxiv.org/abs/1612.06037.

[^2]: Tang, Wei, Zhenwei Shi, and Zhenyu An. “Nonnegative Matrix Factorization for Hyperspectral Unmixing Using Prior Knowledge of Spectral Signatures.” Optical Engineering 51, no. 8 (August 2012): 087001. https://doi.org/10.1117/1.OE.51.8.087001.

[^3]: Ding, Chris H.Q., Tao Li, and Michael I. Jordan. “Convex and Semi-Nonnegative Matrix Factorizations.” IEEE Transactions on Pattern Analysis and Machine Intelligence 32, no. 1 (January 2010): 45–55. https://doi.org/10.1109/TPAMI.2008.277.
