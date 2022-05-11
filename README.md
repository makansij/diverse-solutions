# Diverse Solutions (Archived Jupyter Notebook)

With the last previous-generation of D-Wave 2000Q quantum computer taken offline,
only lower-noise D-Wave 2000Q quantum processing units (QPU) are available now
in [Leap](https://cloud.dwavesys.com/leap/).
The *Diverse Solutions* Jupyter Notebook, which compared the two, has
been archived [here](./diverse-solutions.md) as a webpage that presents the
output of a typical execution.

This notebook makes use of D-Wave's 2019 breakthrough in quantum device
fabrication&mdash;its lower-noise technology for the quantum processing unit
(QPU)&mdash;to provide some tools and demonstrate techniques that can help your
quantum applications find better, more robust solutions to hard problems.

The D-Wave quantum computer solves binary quadratic models (BQM), the Ising model
traditionally used in statistical mechanics and its computer-science equivalent,
the quadratic unconstrained binary optimization (QUBO) problem. These formulations
can express a [wide range](https://arxiv.org/abs/1302.5843) of hard optimization
and constraint satisfaction problems, for example, such as job-shop scheduling,
protein folding, and traffic-flow optimization.

For applications in such fields to be successful may require diverse solutions,
which enable the application to respond to changes in the environment (problem
conditions). Additionally, a diverse set of solutions may reflect and enable
analysis of structural characteristics of the problem.

The notebook has the following sections:

1. **Lower-Noise QPU** gives a basic explanation of what noise
   is reduced in newer-generation QPUs.
2. **Basic Solution Analysis** shows some quick and
   simple ways to examine problem solutions.
3. **Analysis with Hamming Distance** demonstrates
   a simple tool for analyzing solutions.
3. **Analysis with Autocorrelation** explains a
   more powerful tool, autocorrelation, and demonstrates its use through a
   comparison of results between the newer and previous generation of QPUs.

## License

See [LICENSE](LICENSE.md) file.
