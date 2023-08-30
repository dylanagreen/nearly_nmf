# Handling Negative Noise with NMF
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

## H update

To derive the update rule we begin by taking the derivative of the objective with respect to the matrix $H$ and setting it equal to zero:
$$\frac{\partial \chi^2}{\partial H} = W^T (V \circ (X^{\prime} - WH - W_0H_0)) = 0,$$

where I have already dropped the numerical factor of $2$. With some simple, but not exhaustive, algebraic manipulation we can find then that a single optimization step for $H$ is given by

$$\begin{aligned}
0 &= W^T (V \circ (X^{\prime} - WH - W_0H_0)) \\
W^T (V\circ X^{\prime}) &= W^T(V \circ (WH + {W_0H_0})) \\
1 &= \frac{W^T (V\circ X^{\prime})}{W^T(V \circ (WH + {W_0H_0}))}\end{aligned}$$

$$\boxed{H \leftarrow H \circ \frac{W^T (V\circ X^{\prime})}{W^T(V \circ (WH + {W_0H_0}))}}$$

With division applied elementwise. Step 3 holds as long as $H$ is truly a minimizer, with step 4 following from 3 as the step necessary to move $H$ towards that minima. Evidently when $H$ is a minimizer it remains unchanged. For more intuition and a more robust proof of this update rule, see Tang et al. 2012 [^2].

## W update

The $W$ update is derived using the same method as the $H$ update, so I will skip the algebra here.

$$\frac{\partial \chi^2}{\partial W} = (V \circ (X^{\prime} - WH - W_0H_0))H^T = 0$$

$$\boxed{W \leftarrow W \circ \frac{(V\circ X^{\prime})H^T}{(V \circ (WH + {W_0H_0}))H^T}}$$

These two update rules ($W$ and $H$) are independent, and can be used in tandem or individuallty. For fitting data to an already trained set of templates, it is necessary to only update the $H$ vector of coefficients, while holding the templates $W$ fixed without update. For training templates to fit to data, it is necessary to use **both** update rules. The usual methodology is to alternate, and use the updated version of one in the other until convergence, but it is not a requirement to train both in a 1:1 ratio.

For a proof of convergence of these udpate rules see the appendix of Tang et al. 2012[^2], which allows the coefficients of the fixed templates, $H_0$ in this note, to vary and which does not include weighting, but for which the proof still broadly applies to this case.

# Code

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

[^1]: Zhu, Guangtun. “Nonnegative Matrix Factorization (NMF) with Heteroscedastic Uncertainties and Missing Data.” arXiv, December 18, 2016. http://arxiv.org/abs/1612.06037.

[^2]: Tang, Wei, Zhenwei Shi, and Zhenyu An. “Nonnegative Matrix Factorization for Hyperspectral Unmixing Using Prior Knowledge of Spectral Signatures.” Optical Engineering 51, no. 8 (August 2012): 087001. https://doi.org/10.1117/1.OE.51.8.087001.