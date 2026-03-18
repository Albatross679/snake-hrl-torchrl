# Report Structure

- **Notation**
- **1. Introduction**
  only brief introduction
- **2. Solver Backend: Elastica**
  - System Formulation
    include how many equations there are in total for each RL step
  - Elastica Algorithm: Cosserat Rod Dynamics
  - Staggered Grid Discretization
  - External force handling
    - use table
  - General comparision between Elastica and Dismech
    - Use tables
    - Internal external force torques types
  - control method: CPG
- **3. DisMech Backend: Discrete Elastic Rods**
  - 3.1 System Formulation
  - Dismech Algorithm: Discrete Elastic Rods
  - 3.2 Discrete Elastic Rod Discretization
  - 3.3 Implicit Time Integration
  - control method: CPG
- **4. general comparision between elastica and dismech**
  - use table
  - what forces and torques they include and don't include
  - publish year / github star
  - solver difference
  - etc
- **Related Work**
  - 4.1 Neural Surrogates for ODE/PDE
  - RL control for soft robots
- **Data collection**
  - 5.1 Elastica Data Collection Pipeline
  use table to demostrate
- Neural surrogate model
  - State and Action Representation for Elastica
  - architeture
    -computational graph place holder
  - Configurations and hyperparamters
  - periodic-pattern-learning address
  - Results
    table and graph placeholders
  - 5.3 Per-Element CPG Phase Encoding
- RL training with Elastica and Dismech
  - challenge
- RL training with trained Surrogate
  - result
- **7. Discussion**
  - 7.1 Physics Calibration Challenges
  - 7.2 Surrogate Architecture Experiments
  - 7.3 Data Collection Pipeline Challenges
  - rl training challenge
- **8. Conclusion**
- **9. Issue Tracker**
  - 9.1 Physics Calibration Issues
  - 9.2 Training Issues
  - 9.3 TorchRL v0.11 Compatibility Issues
  - 9.4 Performance and System Issues
  - 9.5 Summary
- Reference