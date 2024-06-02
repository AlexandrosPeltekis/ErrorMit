This repository contains work establishing a benchmarking pipeline for various error mitigation techniques implemented by Mitiq. 

Idea for getting going: start by benchmarking zero-noise extrapolation (ZNE), comparing different extrapolation techniques, as well as  probabilistic error cancellation (PEC) 
    - these are the two that seem to be easily implemented in Mitiq already
    - Mitiq can already run many experiments for comparing different “zne” strategies, “scale_noise” methods ,and ZNE factories for extrapolation, for which it computes an 'improvement factor'
    - we may want to try running these types of experiements for increasing number of qubits and increasing circuit depth


Other methods we can think about implementing once we have the infrastructure established from the first two:
     randomization methods, subspace expansion
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GUI Application</title>
</head>
<body>
    <h1>GUI Application</h1>
    <p>This is a graphical user interface (GUI) application built using Python. It provides a user-friendly interface for interacting with the underlying functionality provided by the appBackend module.</p>

    <h2>Prerequisites</h2>
    <p>Before running this application, ensure you have the following dependencies installed:</p>
    <ul>
        <li>Python 3.x</li>
        <li>CustomTkinter</li>
        <li>Matplotlib</li>
        <li>Toml</li>
        <li>Tomlkit</li>
    </ul>
    <p>You can install these dependencies using pip by running:</p>
    <pre>pip install -r requirements.txt</pre>

    <h2>Usage</h2>
    <p>To run the GUI application, execute the following command:</p>
    <pre>python gui.py</pre>
    <p>This will launch the GUI window where you can interact with the application.</p>

    <h2>Features</h2>
    <p>- Describe key features and functionalities of the GUI application.</p>

    <h2>Screenshots</h2>
    <p>- Include screenshots of the GUI application if applicable.</p>

    <h2>License</h2>
    <p>This project is licensed under the MIT License.</p>
</body>
</html>
