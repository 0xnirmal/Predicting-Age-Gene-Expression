# Paper Introduction Rewrite #

For my first paper that targets a broad rather than a specialized audience, I chose the [Modeling Clinical Time Series Using Gaussian Process Sequences paper][1]. The reason I felt this is directed to a broad audience is that it defines a problem, presents several current solutions in the field, and then synthesizes them into its own solution. In this way, the paper assumes no knowledge of the current solutions/models and reads like a textbook. I rewrote the beginning of the introduction with the idea that the reader is familar with linear dynamical systems and gaussian processes and simply need to learn how their new model, a synethesis of these models, solves the domain-specific problems associated with clinical time series modeling. 

_Development of accurate models of clinical time series is extremely important for disease prediction and patient management (taken directly from the original paper).  Current solutions like linear dynamical systems (LDS) and gaussian processes (GP) are not sufficient alone due to domain-specific challenges. These challenges include non-linear trajectories, which the LDS  cannot model and variable length/irregularly sampled data, which the GP can have difficulties modeling. The primary difficulty the GP has is that the mean-function must be time invariant to account for the irregular sampling; however, this can result in poor accuracies on prediction because we are losing degrees of freedom with contant mean. We present a solution, the State-Space Gaussian Process (SSGP), which combines the LDS and GP into a flexible model capable of handling the dynamics of clinical time series data. The model divides the time space into windows, in which transitions between windows are modeled using the LDS, while intra-window modeling is handled by the emissions from the LDS and independently trained GPs._




[1]: https://people.cs.pitt.edu/~milos/research/sdm_zitao_2013.pdf
