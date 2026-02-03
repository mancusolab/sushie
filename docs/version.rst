.. _versions:

=========
Version History
=========

.. list-table::
   :header-rows: 1

   * - Version
     - Description
   * - 0.1
     - Initial Release
   * - 0.11
     - Fix the bug for OLS to compute adjusted r squared.
   * - 0.12
     - Update io.corr function so that report all the correlation results no matter cs is pruned or not.
   * - 0.13
     - Add ``--keep`` command to enable user to specify a file that contains the subjects ID SuShiE will perform on. Add  ``--ancestry_index`` command to enable user to specify a file that contains the ancestry index for fine-mapping. With this, user can input single phenotype, genotype, and covariate file that contains all the subjects across ancestries. Implement padding to increase inference time. Record elbo at each iteration and can access it in the ``infer.SuShiEResult`` object. The alphas table now outputs the average purity and KL divergence for each ``L``. Change ``--kl_threshold`` to ``--divergence``. Add ``--maf`` command to remove SNPs that less than minor allele frequency threshold within each ancestry. Add ``--max_select`` command to randomly select maximum number of SNPs to compute purity to avoid unnecessary memory spending. Add a QC function to remove duplicated SNPs.
   * - 0.14
     - Remove KL-Divergence pruning. Enhance command line appearance and improve the output files contents. Fix small bugs on multivariate KL.
   * - 0.15
     - Fix several typos; add a sanity check on reading vcf genotype data by assigning gt_types==Unknown as NA; Add preprint information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
   * - 0.16
     - Implement summary-level data inference. Add option to remove ambiguous SNPs; fix several bugs and enhance codes quality.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
   * - 0.17
     - Fix several bugs, add debug checkpoints, add chrom, start, and end filtering to individual-level fine-mapping, enhance codes quality, and update readme for official publication.
   * - 0.18
     - Add function that outputs log bayes factor in the alphas file. Update the documentation.
   * - 0.19
     - Add CI/CD pipeline with GitHub Actions (testing across Python 3.9-3.11, linting, type checking, coverage). Improve code quality: fix PEP 8 violations, tighten mypy configuration, improve warning handling. Add troubleshooting section to README and comprehensive FAQ documentation. **This update was completely done using** `Claude Code <https://claude.ai/claude-code>`_.
