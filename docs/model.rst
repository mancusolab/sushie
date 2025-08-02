.. _Model:

=================
Model Description
=================

The sum of single shared effect (SuShiE) extends the sum of single effect (SuSiE) [1]_ model by introducing a prior correlation estimator to account for the ancestral quantitative trait loci (QTL) effect size similarity and differences. Specifically, for :math:`i^{\text{th}}` of total :math:`k \in \mathbb{N}` ancestries, we model the molecular data :math:`g_i \in \mathbb{R}^{n_i \times 1}` for :math:`n_i \in \mathbb{N}` individuals as a linear combination of standardized genotype matrix :math:`X_i \in \mathbb{R}^{n_i \times p}` for :math:`p \in \mathbb{N}` SNPs as

.. math::
   :nowrap:

   \begin{gather*}
   g_i = X_i \beta_i+\epsilon_i \\

   \beta_i = \sum_{l=1}^{L}\beta_{i,l} \\

   \beta_{i,l} = \gamma_l \cdot b_{i, l} \\

   b_{l} = \begin{bmatrix} b_{1,l} \\ \vdots \\ b_{k,l} \end{bmatrix} \sim \mathcal{N}(0, C_l) \\

   C_{i,i',l} = \begin{cases} \sigma_{i,b,l}^2 & \text{if } i = i' \\ \rho_{i,i',l} \cdot \sigma_{i,b,l} \cdot \sigma_{i',b,l} & \text{otherwise}\end{cases} \\

   \gamma_l   \sim \text{Multi}(1, \pi) \\

   \epsilon_i \sim \mathcal{N}(0, \sigma^2_{i, e}I_{n_i}) \\
   \end{gather*}

where :math:`\beta_i \in \mathbb{R}^{p \times1}` is the shared QTL effects, :math:`\epsilon_i \in \mathbb{R}^{n_i \times 1}` is the ancestry-specific effects and other environmental noises, :math:`L \in \mathbb{R}` is the number of shared effects, for  :math:`l^{\text{th}}`  single shared effect,  :math:`b_{i,l} \in \mathbb{R}` is a scaler representing effect size, :math:`C_l \in \mathbb{R}^{k \times k}` is the prior covariance matrix with :math:`\sigma^2_{i,b}` as variance and :math:`\rho` as correlation, :math:`\gamma_l` is an binary indicator vector specifying which single SNP is the QTL, :math:`\pi` is the prior probability for each SNP to be QTL, and :math:`\sigma^2_e` is the prior variance for noises.

SuShiE runs `varitional inference <https://en.wikipedia.org/wiki/Variational_Bayesian_methods>`_ to estimate the posterior distribution for :math:`\beta_l` and :math:`\gamma_l` for each :math:`l^{\text{th}}` effect. We can quantify the probability of QTL for each SNP through Posterior Inclusion Probabilities (PIPs). If the posterior distribution of :math:`\gamma_l` is :math:`\text{Multi}(1, \alpha_l)`, then for each SNP :math:`j`, we have:

.. math::
   \text{PIP}_j = 1 - \prod_{l=1}^L(1 - \alpha_{l, j})

For more details in math derivation and algorithm, stay tuned for our upcoming manuscript.

.. _Reference:

Reference
==========
.. [1] Wang, G., Sarkar, A., Carbonetto, P. and Stephens, M. (2020), A simple new approach to variable selection in regression, with application to genetic fine mapping. J. R. Stat. Soc. B, 82: 1273-1300. https://doi.org/10.1111/rssb.12388
