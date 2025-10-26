# Quantum Random Number Generator (QRNG) Project

A comprehensive quantum random number generation project featuring both a command-line terminal application and an interactive web-based Streamlit application. This project leverages quantum superposition principles using Qiskit to produce truly random numbers, unlike classical pseudo-random number generators.

## Description

This QRNG project demonstrates quantum computing applications by using Hadamard gates to create superposition states in qubits. When measured, these states collapse into random classical bits, providing true randomness that cannot be predicted even with infinite computational power.

The project includes two implementations:

1. **qrng.py**: A command-line Python script for generating quantum random numbers with statistical analysis and visualization.
2. **app.py**: An interactive Streamlit web application offering multiple demos including random number generation, secure financial transactions, and cryptocurrency applications.

Both implementations use Qiskit for quantum circuit simulation and provide statistical validation of randomness quality.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Dependencies for qrng.py (Terminal Version)
Install the required packages for the command-line application:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib
```

### Dependencies for app.py (Web Application)
Install the additional packages required for the Streamlit web app:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib streamlit pandas
```

You can install all dependencies at once:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib streamlit pandas
```

## Usage

### Running qrng.py (Terminal Version)

Run the command-line application from your terminal:

```bash
python qrng.py
```

The application will prompt you for the following parameters:

1. **Number of qubits** (1-10): Determines the range of random numbers (0 to 2^n - 1)
2. **Number of shots**: How many times to run the quantum circuit (recommended: 100-10000)
3. **Optimization level** (0-3): Transpiler optimization for the quantum circuit

#### Example Output for qrng.py

```
======================================================================
  QUANTUM RANDOM NUMBER GENERATOR
======================================================================

Please enter the following parameters:
Number of qubits (1-10): 3
Number of shots (positive integer): 1000
Optimization level (0-3): 2

Configuration:
   Number of Qubits:      3
   Number of Shots:       1000
   Optimization Level:    2
   Range of Outcomes:     0 to 7

[Quantum Circuit Creation and Simulation Output...]

Generated Random Number:
──────────────────────────────────────────────────────────────────────
  Decimal: 4
  Binary:  100
  Hex:     4

[Statistical Analysis and Visualization Output...]
```

### Running app.py (Web Application)

Launch the interactive Streamlit web application:

```bash
streamlit run app.py
```

This will start a local web server (typically at http://localhost:8501). Open this URL in your web browser to access the application.

The web app provides three main demo modes:

1. **Random Number Generation**: Interactive quantum RNG with configurable parameters
2. **Secure Financial Transactions**: Demo of quantum-generated encryption keys for banking
3. **Cryptocurrency & Blockchain**: Applications in wallet generation, transaction nonces, and mining

## Features

### qrng.py (Terminal Version) Features

#### Quantum Circuit Generation
- Creates quantum circuits with configurable number of qubits (1-10)
- Applies Hadamard gates to create equal superposition states
- Adds measurement operations on all qubits

#### Statistical Analysis
- **Shannon Entropy**: Measures the quality of randomness (closer to theoretical maximum = better)
- **Chi-Square Uniformity Test**: Tests if distribution matches uniform randomness
- Probability calculations for each possible outcome

#### Visualization
- Generates and saves distribution plots as PNG images
- Terminal-based histogram display
- Comparison with expected uniform distribution

#### Random Number Generation
- Generates a single random number from quantum measurements
- Displays in decimal, binary, and hexadecimal formats
- Provides access to all generated random samples

#### Configuration Options
- User input validation for all parameters
- Configurable optimization levels for circuit transpilation
- Error handling and informative output

### app.py (Web Application) Features

#### Interactive Configuration
- Slider and number inputs for qubits, shots, and optimization
- Real-time parameter validation
- Display options (show bitstrings, enable downloads, etc.)

#### Multiple Demo Modes
- **Random Number Generation**: Core QRNG functionality with live visualization
- **Secure Financial Transactions**: Quantum key generation for transaction encryption
- **Cryptocurrency & Blockchain**: Wallet addresses, transaction nonces, mining simulation

#### Advanced Visualizations
- Interactive matplotlib plots with quantum vs. classical comparisons
- Real-time distribution histograms
- Statistical analysis dashboards

#### Statistical Analysis Dashboard
- Live Shannon entropy calculations
- Chi-square uniformity testing
- Probability distribution analysis

#### Export and Integration Features
- CSV download of results
- Cryptographic hash generation (SHA-256)
- Encryption demonstrations

#### Security Demonstrations
- Quantum-generated encryption keys (128-512 bits)
- XOR encryption simulation
- Transaction security flow visualization

## How It Works

### Quantum Principles
1. **Superposition**: Hadamard gates transform qubits from |0⟩ to (|0⟩ + |1⟩)/√2, creating equal probability for 0 and 1
2. **Measurement Collapse**: When measured, the superposition randomly collapses to either 0 or 1
3. **True Randomness**: Quantum measurement outcomes are fundamentally unpredictable

### Technical Implementation
1. **Circuit Creation**: Quantum circuits are built using Qiskit with n qubits and n classical bits
2. **Simulation**: Circuits are executed on the AerSimulator backend multiple times ("shots")
3. **Result Processing**: Measurement counts are converted to probabilities and statistical measures
4. **Analysis**: Entropy and uniformity tests validate randomness quality
5. **Visualization**: Results are plotted and analyzed for distribution uniformity

### Key Differences Between Implementations
- **qrng.py**: Focuses on detailed terminal output, statistical analysis, and file-based visualization
- **app.py**: Provides interactive GUI, multiple application demos, and real-time feedback

## Applications

### Cryptography and Security
- Generation of truly random encryption keys
- Secure nonce creation for authentication
- Quantum-resistant cryptographic primitives

### Scientific Research
- Monte Carlo simulations requiring unbiased randomness
- Statistical sampling and hypothesis testing
- Random matrix generation

### Gaming and Gambling
- Fair random number generation for lotteries
- Unpredictable game outcomes
- Secure random seed generation

### Financial Technology
- Secure transaction ID generation
- Quantum-enhanced banking security
- Cryptocurrency wallet security

### Blockchain and Cryptocurrency
- Wallet address generation
- Transaction nonce creation
- Mining nonce randomization

## Technical Details

- **Quantum Backend**: Qiskit AerSimulator for noiseless quantum simulation
- **Optimization**: Configurable transpiler optimization levels (0-3)
- **Performance**: Execution time scales with qubits and shots
- **Limitations**: Maximum ~10 qubits for qrng.py, ~28 qubits for app.py due to computational constraints
- **Dependencies**: Qiskit ecosystem, scientific Python stack, Streamlit for web interface

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install --upgrade qiskit qiskit-aer numpy scipy matplotlib streamlit pandas
   ```

2. **Memory Issues**: Reduce shots or qubits for large simulations

3. **Web App Not Loading**: Check that Streamlit is installed and port 8501 is available

4. **Plot Not Saving**: Ensure write permissions in the current directory

### Performance Tips

- Start with 3-5 qubits and 100-1000 shots for testing
- Use optimization level 2 for balanced performance
- Increase shots for better statistical accuracy
- For web app, use fewer shots initially for faster response

## License

This project is developed by Team LOCO MINDS. See individual file headers for licensing information.

## Contributing

Contributions are welcome! Please ensure code follows existing style and includes appropriate documentation.

## Authors

Team LOCO MINDS

## Related Projects

- [Qiskit](https://qiskit.org/) - Quantum computing framework
- [Streamlit](https://streamlit.io/) - Web app framework
- [Qiskit Aer](https://qiskit.org/ecosystem/aer/) - High-performance quantum simulation
