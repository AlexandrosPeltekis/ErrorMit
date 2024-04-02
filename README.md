This repository contains work establishing a benchmarking pipeline for various error mitigation techniques implemented by Mitiq. 

Idea for getting going: start by benchmarking zero-noise extrapolation (ZNE), comparing different extrapolation techniques, as well as  probabilistic error cancellation (PEC) 
    - these are the two that seem to be easily implemented in Mitiq already
    - Mitiq can already run many experiments for comparing different “zne” strategies, “scale_noise” methods ,and ZNE factories for extrapolation, for which it computes an 'improvement factor'
    - we may want to try running these types of experiements for increasing number of qubits and increasing circuit depth


Other methods we can think about implementing once we have the infrastructure established from the first two:
     randomization methods, subspace expansion