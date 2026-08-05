# Model description {#model}

The sum of single shared effect (SuShiE) extends the sum of single effect (SuSiE)[^1] model by introducing a prior correlation estimator to account for the ancestral quantitative trait loci (QTL) effect size similarity and differences. Specifically, for $i^{\text{th}}$ of total $k \in \mathbb{N}$ ancestries, we model the molecular data $g_i \in \mathbb{R}^{n_i \times 1}$ for $n_i \in \mathbb{N}$ individuals as a linear combination of standardized genotype matrix $X_i \in \mathbb{R}^{n_i \times p}$ for $p \in \mathbb{N}$ SNPs as

$$
\begin{aligned}
g_i &= X_i \beta_i + \epsilon_i \\
\beta_i &= \sum_{l=1}^{L}\beta_{i,l} \\
\beta_{i,l} &= \gamma_l \cdot b_{i,l} \\
b_l &= \begin{bmatrix} b_{1,l} \\ \vdots \\ b_{k,l} \end{bmatrix}
      \sim \mathcal{N}(0, C_l) \\
C_{i,i',l} &=
\begin{cases}
\sigma_{i,b,l}^2 & \text{if } i = i' \\
\rho_{i,i',l} \cdot \sigma_{i,b,l} \cdot \sigma_{i',b,l} & \text{otherwise}
\end{cases} \\
\gamma_l &\sim \operatorname{Multi}(1, \pi) \\
\epsilon_i &\sim \mathcal{N}(0, \sigma^2_{i,e}I_{n_i})
\end{aligned}
$$

where $\beta_i \in \mathbb{R}^{p \times1}$ is the shared QTL effects, $\epsilon_i \in \mathbb{R}^{n_i \times 1}$ is the ancestry-specific effects and other environmental noises, $L \in \mathbb{R}$ is the number of shared effects, for $l^{\text{th}}$ single shared effect, $b_{i,l} \in \mathbb{R}$ is a scaler representing effect size, $C_l \in \mathbb{R}^{k \times k}$ is the prior covariance matrix with $\sigma^2_{i,b}$ as variance and $\rho$ as correlation, $\gamma_l$ is an binary indicator vector specifying which single SNP is the QTL, $\pi$ is the prior probability for each SNP to be QTL, and $\sigma^2_e$ is the prior variance for noises.

SuShiE runs [variational inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) to estimate the posterior distribution for $\beta_l$ and $\gamma_l$ for each $l^{\text{th}}$ effect. We can quantify the probability of QTL for each SNP through Posterior Inclusion Probabilities (PIPs). If the posterior distribution of $\gamma_l$ is $\text{Multi}(1, \alpha_l)$, then for each SNP $j$, we have:

$$\text{PIP}_j = 1 - \prod_{l=1}^L(1 - \alpha_{l, j})$$

For more details on the mathematical derivation and algorithm, see our [Nature Genetics paper](https://www.nature.com/articles/s41588-025-02262-7).

## Reference

[^1]: Wang, G., Sarkar, A., Carbonetto, P. and Stephens, M. (2020), A simple new approach to variable selection in regression, with application to genetic fine mapping. J. R. Stat. Soc. B, 82: 1273-1300. <https://doi.org/10.1111/rssb.12388>
