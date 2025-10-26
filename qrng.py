import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Default parameters (will be overridden by user input)
N_QUBITS = 5                # Number of qubits (1-10)
SHOTS = 1000                # Number of measurements
OPTIMIZATION_LEVEL = 2     # Transpiler optimization (0-3)

# HELPER FUNCTIONS

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_section(text):
    """Print a formatted section header."""
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print(f"{'─'*70}")

def create_qrng_circuit(n_qubits):
   
    # Create quantum circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Apply Hadamard gate to each qubit (creates superposition)
    for i in range(n_qubits):
        qc.h(i)
    
    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc

def run_quantum_simulation(circuit, shots, optimization_level):
   
    # Initialize simulator
    simulator = AerSimulator()
    
    # Transpile circuit
    transpiled_circuit = transpile(
        circuit, 
        simulator, 
        optimization_level=optimization_level
    )
    
    # Run simulation
    start_time = time.time()
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    execution_time = time.time() - start_time
    
    # Get counts
    counts = result.get_counts()
    
    return counts, execution_time

def process_results(counts, n_qubits, shots):
    
    max_outcome = 2 ** n_qubits
    
    # Initialize all possible outcomes with 0 counts
    results = {}
    for i in range(max_outcome):
        bitstring = format(i, f'0{n_qubits}b')
        results[i] = {
            'bitstring': bitstring,
            'count': 0,
            'probability': 0.0
        }
    
    # Fill in actual counts
    for bitstring, count in counts.items():
        outcome_int = int(bitstring, 2)
        results[outcome_int]['count'] = count
        results[outcome_int]['probability'] = count / shots
    
    return results

def calculate_shannon_entropy(probabilities):
    
    p = np.array([prob for prob in probabilities if prob > 0])
    entropy = -np.sum(p * np.log2(p))
    return entropy

def chi_square_uniformity_test(observed_counts, expected_count):
    
    observed = np.array(observed_counts)
    expected = np.full_like(observed, expected_count, dtype=float)
    
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    return chi2_stat, p_value

def print_results_table(results, n_qubits, shots):
    """Print formatted results table."""
    print(f"\n{'Outcome':<10} {'Bitstring':<15} {'Count':<10} {'Probability':<15}")
    print("─" * 50)
    
    for outcome in sorted(results.keys()):
        data = results[outcome]
        print(f"{outcome:<10} {data['bitstring']:<15} {data['count']:<10} {data['probability']:<15.4f}")
    
    print(f"\n{'Total:':<25} {shots:<10}")

def plot_distribution(results, n_qubits, shots):
    
    # Use non-interactive backend for terminal
    import matplotlib
    matplotlib.use('Agg')

    outcomes = sorted(results.keys())
    counts = [results[o]['count'] for o in outcomes]

    # Calculate expected uniform count
    expected_count = shots / len(outcomes)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot bars
    bars = plt.bar(outcomes, counts, color='#1f77b4', alpha=0.7,
                   edgecolor='black', linewidth=0.8)

    # Add expected uniform line
    plt.axhline(y=expected_count, color='red', linestyle='--',
                linewidth=2, label=f'Expected (Uniform): {expected_count:.1f}')

    # Formatting
    plt.xlabel('Outcome (Integer)', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(f'Quantum Random Number Distribution\n({n_qubits} qubits, {shots} shots)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    # Set x-axis ticks
    if len(outcomes) <= 16:
        plt.xticks(outcomes)
    else:
        tick_step = max(1, len(outcomes) // 16)
        plt.xticks(range(0, len(outcomes), tick_step))

    plt.tight_layout()

    # Save plot
    filename = f'qrng_distribution_{n_qubits}qubits_{shots}shots.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {filename}")

    # Close the figure to free memory
    plt.close()

def plot_distribution_terminal(results, n_qubits, shots):
    
    print("\nDistribution Histogram (Terminal):")
    print("-" * 50)

    outcomes = sorted(results.keys())
    counts = [results[o]['count'] for o in outcomes]
    max_count = max(counts) if counts else 1

    for outcome, count in zip(outcomes, counts):
        # Create a simple bar using asterisks
        bar_length = int((count / max_count) * 40) if max_count > 0 else 0
        bar = '*' * bar_length
        percentage = (count / shots) * 100 if shots > 0 else 0
        print(f"{outcome:2d}: {bar:<40} {count:4d} ({percentage:5.1f}%)")

    # Expected uniform count
    expected_count = shots / len(outcomes)
    print(f"\nExpected uniform count: {expected_count:.1f}")
    print(f"Total outcomes: {len(outcomes)}")
    print(f"Total shots: {shots}")

def print_statistical_analysis(results, n_qubits, shots):
    """Print statistical analysis results."""
    print_section("STATISTICAL ANALYSIS")
    
    # Extract probabilities and counts
    probabilities = [results[o]['probability'] for o in sorted(results.keys())]
    counts = [results[o]['count'] for o in sorted(results.keys())]
    
    # Shannon Entropy
    entropy = calculate_shannon_entropy(probabilities)
    theoretical_max = n_qubits
    entropy_ratio = (entropy / theoretical_max) * 100
    
    print("\nShannon Entropy:")
    print(f"   Measured Entropy:      {entropy:.4f} bits")
    print(f"   Theoretical Maximum:   {theoretical_max} bits")
    print(f"   Entropy Ratio:         {entropy_ratio:.2f}%")

    if entropy > theoretical_max * 0.95:
        print("   Excellent randomness - entropy very close to maximum!")
    elif entropy > theoretical_max * 0.85:
        print("   Good randomness - entropy is reasonably high")
    else:
        print("   Lower entropy - consider increasing shots")
    
    # Chi-Square Test
    expected_count = shots / len(counts)
    chi2_stat, p_value = chi_square_uniformity_test(counts, expected_count)
    
    print("\nChi-Square Uniformity Test:")
    print(f"   χ² Statistic:          {chi2_stat:.4f}")
    print(f"   p-value:               {p_value:.4f}")
    print(f"   Degrees of Freedom:    {len(counts) - 1}")

    if p_value >= 0.01:
        print("   Distribution consistent with uniformity (p ≥ 0.01)")
    else:
        print("   Distribution may deviate from uniformity (p < 0.01)")
    
    print("\nInterpretation:")
    print("   A high p-value (≥ 0.01) suggests the distribution is statistically")
    print("   consistent with a uniform distribution, indicating good quantum randomness.")

def generate_random_numbers(results, n_samples=10):
    
    # Create weighted list based on counts
    random_numbers = []
    for outcome, data in results.items():
        random_numbers.extend([outcome] * data['count'])
    
    # Shuffle to randomize order
    np.random.shuffle(random_numbers)
    
    return random_numbers[:n_samples]

# MAIN EXECUTION

def main():
    """Main execution function."""

    # Print header
    print_header("QUANTUM RANDOM NUMBER GENERATOR")

    # Get user input for parameters
    print("\nPlease enter the following parameters:")

    # Input validation loop for qubits
    while True:
        try:
            n_qubits_input = input("Number of qubits (1-10): ").strip()
            n_qubits = int(n_qubits_input)
            if 1 <= n_qubits <= 10:
                break
            else:
                print("Error: Number of qubits must be between 1 and 10.")
        except ValueError:
            print("Error: Please enter a valid integer.")

    # Input validation loop for shots
    while True:
        try:
            shots_input = input("Number of shots (positive integer): ").strip()
            shots = int(shots_input)
            if shots > 0:
                break
            else:
                print("Error: Number of shots must be a positive integer.")
        except ValueError:
            print("Error: Please enter a valid integer.")

    # Input validation loop for optimization level
    while True:
        try:
            opt_input = input("Optimization level (0-3): ").strip()
            optimization_level = int(opt_input)
            if 0 <= optimization_level <= 3:
                break
            else:
                print("Error: Optimization level must be between 0 and 3.")
        except ValueError:
            print("Error: Please enter a valid integer.")

    # Set global variables based on user input
    global N_QUBITS, SHOTS, OPTIMIZATION_LEVEL
    N_QUBITS = n_qubits
    SHOTS = shots
    OPTIMIZATION_LEVEL = optimization_level

    # Configuration summary
    print("\nConfiguration:")
    print(f"   Number of Qubits:      {N_QUBITS}")
    print(f"   Number of Shots:       {SHOTS}")
    print(f"   Optimization Level:    {OPTIMIZATION_LEVEL}")
    print(f"   Range of Outcomes:     0 to {2**N_QUBITS - 1}")

    # STEP 1: CREATE QUANTUM CIRCUIT

    print_section("QUANTUM CIRCUIT")
    
    print("\nCreating quantum circuit...")
    qc = create_qrng_circuit(N_QUBITS)
    
    print("\nCircuit Diagram:")
    print(qc.draw(output='text'))

    print(f"\nCircuit created with {N_QUBITS} qubits")
    print(f"   - Applied Hadamard gates to all qubits (superposition)")
    print(f"   - Added measurements to all qubits")

    # STEP 2: RUN QUANTUM SIMULATION
   
    print_section("QUANTUM SIMULATION")
    
    print(f"\nRunning {SHOTS} shots on AerSimulator...")
    counts, exec_time = run_quantum_simulation(qc, SHOTS, OPTIMIZATION_LEVEL)
    
    print(f"\nSimulation completed!")
    print(f"   Execution Time:  {exec_time:.3f} seconds")
    print(f"   Unique Outcomes: {len(counts)}")

    # STEP 3: PROCESS RESULTS

    print_section("RESULTS")
    
    print("\nProcessing measurement results...")
    results = process_results(counts, N_QUBITS, SHOTS)
    
    # Print results table
    print_results_table(results, N_QUBITS, SHOTS)
    
    # STEP 4: GENERATE RANDOM NUMBER

    print_section("RANDOM NUMBER GENERATED")

    random_samples = generate_random_numbers(results, n_samples=1)
    num = random_samples[0]

    binary = format(num, f'0{N_QUBITS}b')
    hex_val = format(num, f'0{(N_QUBITS+3)//4}x')

    print("\nGenerated Random Number:")
    print("─" * 70)
    print(f"  Decimal: {num}")
    print(f"  Binary:  {binary}")
    print(f"  Hex:     {hex_val}")

    print(f"\nTotal random numbers available: {SHOTS}")
    
    # STEP 5: STATISTICAL ANALYSIS

    print_statistical_analysis(results, N_QUBITS, SHOTS)

    # STEP 6: VISUALIZATION
    
    print_section("VISUALIZATION")

    # Generate and save image plot
    print("\nGenerating distribution plot image...")
    plot_distribution(results, N_QUBITS, SHOTS)

    # SUMMARY

    print_section("SUMMARY")
    
    print(f"""
Quantum Random Number Generation Complete!

   Generated Range:       0 to {2**N_QUBITS - 1}
   Total Measurements:    {SHOTS}
   Unique Outcomes:       {len([r for r in results.values() if r['count'] > 0])} / {2**N_QUBITS}
   Execution Time:        {exec_time:.3f} seconds

   The quantum circuit used Hadamard gates to create superposition,
   ensuring true quantum randomness from measurement collapse.
    """)
    
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()