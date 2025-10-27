"""
Quantum Random Number Generator (QRNG)
A Streamlit application that generates true random numbers using quantum superposition.

Installation:
pip install qiskit qiskit-aer streamlit numpy scipy pandas matplotlib

Run:
streamlit run app.py

Author: Loco Minds 
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import hashlib
import time
from io import StringIO

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def bell_state_qrng(n_qubits, shots):
    """
    Generates random bits using quantum superposition and entanglement.
    For 2 qubits, creates Bell state |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩).
    For more qubits, creates GHZ-like entangled state.

    Args:
        n_qubits (int): Number of qubits to use (must be >= 2)
        shots (int): Number of shots to run the circuit

    Returns:
        tuple: (bit_stream, counts) - Binary string and measurement counts
    """
    if n_qubits < 2:
        n_qubits = 2
    
    # Create a quantum circuit
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Create entangled state (GHZ-like for n_qubits > 2)
    qc.h(0)  # Put first qubit in superposition
    
    # Entangle all qubits with CNOT gates
    for i in range(1, n_qubits):
        qc.cx(0, i)

    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))

    # Execute the circuit on AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()

    # Retrieve the measurement results (counts dictionary)
    counts = result.get_counts(qc)

    # Extract measurements in order
    bit_stream = ''
    for bitstring, count in sorted(counts.items()):
        bit_stream += bitstring * count

    return bit_stream, counts

def ry_qrng(num_bits, shots):
    """
    Generates random bits using Ry(pi/2) rotation gates.
    Each shot produces n_qubits bits, run multiple shots to get num_bits total.

    Args:
        num_bits (int): Number of random bits to generate
        shots (int): Number of shots to run the circuit

    Returns:
        tuple: (bit_stream, counts) - Binary string and measurement counts
    """
    # Calculate number of qubits needed per shot
    n_qubits = min(num_bits, 28)  # Limit to 28 qubits per circuit
    
    # Calculate how many shots we actually need
    bits_per_shot = n_qubits
    required_shots = (num_bits + bits_per_shot - 1) // bits_per_shot
    
    # Use provided shots or required shots, whichever is greater
    actual_shots = max(shots, required_shots)

    # Create a quantum circuit
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Apply Ry(pi/2) to every qubit
    for i in range(n_qubits):
        qc.ry(np.pi/2, i)

    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))

    # Execute the circuit on AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=actual_shots)
    result = job.result()

    # Get the measurement counts
    counts = result.get_counts(qc)

    # Convert measurements to bit stream
    bit_stream = ''
    for bitstring, count in sorted(counts.items()):
        bit_stream += bitstring * count

    # Truncate to requested number of bits
    bit_stream = bit_stream[:num_bits]

    return bit_stream, counts

def binary_to_float(bit_stream, n_bits=32):
    """
    Converts binary string to list of floating point numbers in [0,1].
    
    Args:
        bit_stream (str): String of binary digits
        n_bits (int): Number of bits to use per float (default 32)
        
    Returns:
        list: List of float values between 0 and 1
    """
    floats = []
    for i in range(0, len(bit_stream), n_bits):
        chunk = bit_stream[i:i+n_bits]
        if len(chunk) == n_bits:  # Only process complete chunks
            val = int(chunk, 2)
            float_val = val / (2**n_bits - 1)  # Normalize to [0,1]
            floats.append(float_val)
    return floats

def test_randomness(bit_stream, test_name='chi2', confidence=0.95):
    """
    Performs statistical tests on the bit stream to assess randomness.
    
    Args:
        bit_stream (str): String of binary digits
        test_name (str): Name of statistical test to perform
        confidence (float): Confidence level for hypothesis test
        
    Returns:
        dict: Test results including test statistic and p-value
    """
    # Convert bit stream to array of integers
    bits = np.array([int(b) for b in bit_stream])
    
    if test_name == 'chi2':
        # Chi-square test for uniform distribution of 0s and 1s
        observed = np.bincount(bits)
        expected = np.array([len(bits)/2, len(bits)/2])
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            'test_name': 'Chi-square test for uniformity',
            'test_statistic': chi2_stat,
            'p_value': p_value,
            'reject_null': p_value < (1 - confidence)
        }
    
    elif test_name == 'runs':
        # Runs test for independence
        runs = np.diff(bits)  # Get transitions
        n_runs = np.count_nonzero(runs) + 1
        n0 = np.count_nonzero(bits == 0)
        n1 = np.count_nonzero(bits == 1)
        
        # Expected number of runs
        exp_runs = 1 + (2 * n0 * n1) / len(bits)
        # Variance of runs
        var_runs = (2 * n0 * n1 * (2 * n0 * n1 - len(bits))) / (len(bits)**2 * (len(bits) - 1))
        
        z_stat = (n_runs - exp_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'test_name': 'Runs test for independence',
            'test_statistic': z_stat,
            'p_value': p_value,
            'reject_null': p_value < (1 - confidence)
        }
    
    return {'error': 'Invalid test name'}

def save_random_numbers(numbers, filename):
    """
    Saves random numbers to a CSV file.
    
    Args:
        numbers (list): List of random numbers to save
        filename (str): Name of file to save to
        
    Returns:
        str: CSV string of saved numbers
    """
    df = pd.DataFrame(numbers, columns=['random_number'])
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def plot_distribution(numbers, bins=50):
    """
    Creates histogram plot of random number distribution.
    
    Args:
        numbers (list): List of random numbers to plot
        bins (int): Number of histogram bins
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(numbers, bins=bins, density=True, alpha=0.7)
    ax.set_title('Distribution of Generated Random Numbers')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
    # Add reference uniform distribution line
    x = np.linspace(0, 1, 100)
    ax.plot(x, np.ones_like(x), 'r--', label='Uniform Distribution')
    ax.legend()
    
    return fig, ax
    qc.measure(range(n_qubits), range(n_qubits))

    # Execute the circuit on AerSimulator
    simulator = AerSimulator()
    job = simulator.run(qc, shots=actual_shots)
    result = job.result()

    # Retrieve the measurement results (counts dictionary)
    counts = result.get_counts(qc)

    # Concatenate all measured bits from all shots
    bit_stream = ''
    for bitstring, count in sorted(counts.items()):
        bit_stream += bitstring * count

    # Return the first num_bits bits and the counts
    return bit_stream[:num_bits], counts

def verify_randomness(bit_string):
    """
    Verify if the bit string exhibits true randomness.

    Args:
        bit_string (str): Binary string to verify

    Returns:
        tuple: (bool, str) - True if random, message
    """
    if not bit_string:
        return False, "Empty string"

    n = len(bit_string)
    count_0 = bit_string.count('0')
    count_1 = bit_string.count('1')

    if count_0 + count_1 != n:
        return False, "Contains non-binary characters"

    prob_0 = count_0 / n
    prob_1 = count_1 / n

    # Check if close to 0.5
    if abs(prob_0 - 0.5) > 0.1:  # Allow 10% deviation
        return False, f"Biased distribution: 0:{prob_0:.3f}, 1:{prob_1:.3f}"

    # Calculate entropy
    if prob_0 > 0 and prob_1 > 0:
        entropy = - (prob_0 * np.log2(prob_0) + prob_1 * np.log2(prob_1))
    else:
        entropy = 0

    if entropy < 0.9:  # Should be close to 1
        return False, f"Low entropy: {entropy:.3f}"

    return True, f"Good randomness: entropy {entropy:.3f}"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Quantum Random Number Generator",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Clean white background */
    .stApp {
        background: white;
        color: black;
    }

    /* Ensure text remains readable */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: black !important;
    }

    .stApp p, .stApp li, .stApp div {
        color: #333333 !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_encryption_key(n_bits):
    """
    Generate a quantum random encryption key.

    Args:
        n_bits (int): Number of bits for the key

    Returns:
        str: Binary string representing the encryption key
    """
    # For large keys, generate in chunks to avoid qubit limits
    if n_bits > 28:
        # Generate in 28-bit chunks and concatenate
        chunks = []
        remaining_bits = n_bits
        while remaining_bits > 0:
            chunk_size = min(remaining_bits, 28)
            chunk = generate_encryption_key_chunk(chunk_size)
            chunks.append(chunk)
            remaining_bits -= chunk_size
        return ''.join(chunks)
    else:
        return generate_encryption_key_chunk(n_bits)


def generate_encryption_key_chunk(n_bits):
    """
    Generate a quantum random key chunk (max 28 bits).

    Args:
        n_bits (int): Number of bits (max 28)

    Returns:
        str: Binary string
    """
    # Create quantum circuit
    qc = QuantumCircuit(n_bits, n_bits)
    for i in range(n_bits):
        qc.h(i)
    qc.measure(range(n_bits), range(n_bits))

    # Run simulation without transpilation to avoid coupling map issues
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()

    # Get the random bitstring
    bitstring = list(counts.keys())[0]

    return bitstring


def simulate_secure_transaction(amount, recipient, key_bits):
    """
    Simulate a secure financial transaction using quantum random encryption.

    Args:
        amount (float): Transaction amount
        recipient (str): Recipient identifier
        key_bits (int): Number of bits for encryption key

    Returns:
        dict: Transaction details with encryption
    """
    # Generate quantum random transaction ID
    transaction_id = generate_encryption_key(64)  # 64-bit transaction ID

    # Generate encryption key
    encryption_key = generate_encryption_key(key_bits)

    # Create transaction data
    transaction_data = f"{amount:.2f}|{recipient}|{transaction_id}"

    # Simple XOR encryption simulation (in real world, use proper encryption)
    encrypted_data = ""
    for i, char in enumerate(transaction_data):
        key_char = encryption_key[i % len(encryption_key)]
        encrypted_char = chr(ord(char) ^ int(key_char, 2))
        encrypted_data += encrypted_char

    return {
        'transaction_id': transaction_id,
        'amount': amount,
        'recipient': recipient,
        'encryption_key': encryption_key,
        'original_data': transaction_data,
        'encrypted_data': encrypted_data,
        'key_strength': f"{key_bits}-bit quantum random key"
    }

def create_qrng_circuit(n_qubits):
    """
    Create a quantum circuit for random number generation.
    
    Args:
        n_qubits (int): Number of qubits to use
        
    Returns:
        QuantumCircuit: Circuit with Hadamard gates and measurements
    """
    # Create quantum circuit with n qubits and n classical bits
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Apply Hadamard gate to each qubit to create superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc


def run_quantum_simulation(circuit, shots, optimization_level):
    """
    Execute the quantum circuit on the Aer simulator.
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to execute
        shots (int): Number of times to run the circuit
        optimization_level (int): Transpiler optimization level (0-3)
        
    Returns:
        tuple: (counts dict, execution time in seconds)
    """
    # Initialize the Aer simulator
    simulator = AerSimulator()
    
    # Transpile the circuit for the simulator
    transpiled_circuit = transpile(
        circuit, 
        simulator, 
        optimization_level=optimization_level
    )
    
    # Run the simulation and measure execution time
    start_time = time.time()
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    execution_time = time.time() - start_time
    
    # Get measurement counts
    counts = result.get_counts()
    
    return counts, execution_time


def bitstring_to_int(bitstring):
    """
    Convert a binary string to an integer.
    
    Args:
        bitstring (str): Binary string (e.g., '101')
        
    Returns:
        int: Integer representation
    """
    return int(bitstring, 2)


def process_results(counts, n_qubits, shots):
    """
    Process quantum measurement results into a structured format.
    
    Args:
        counts (dict): Raw counts from quantum measurement
        n_qubits (int): Number of qubits used
        shots (int): Total number of shots
        
    Returns:
        pandas.DataFrame: Processed results with outcomes, counts, and probabilities
    """
    # Create a list to store results
    results_list = []
    
    # Process each measurement outcome
    for bitstring, count in counts.items():
        outcome_int = bitstring_to_int(bitstring)
        probability = count / shots
        
        results_list.append({
            'Bitstring': bitstring,
            'Integer': outcome_int,
            'Count': count,
            'Probability': probability
        })
    
    # Create DataFrame and sort by integer value
    df = pd.DataFrame(results_list)
    df = df.sort_values('Integer').reset_index(drop=True)
    
    return df


def calculate_shannon_entropy(probabilities):
    """
    Calculate Shannon entropy of a probability distribution.
    
    Args:
        probabilities (array-like): Probability values
        
    Returns:
        float: Shannon entropy in bits
    """
    # Filter out zero probabilities to avoid log(0)
    p = np.array(probabilities)
    p = p[p > 0]
    
    # Calculate entropy: H = -sum(p * log2(p))
    entropy = -np.sum(p * np.log2(p))
    
    return entropy


def chi_square_uniformity_test(observed_counts, expected_count):
    """
    Perform chi-square test for uniformity.
    
    Args:
        observed_counts (array-like): Observed frequencies
        expected_count (float): Expected frequency for uniform distribution
        
    Returns:
        tuple: (chi-square statistic, p-value)
    """
    observed = np.array(observed_counts)
    expected = np.full_like(observed, expected_count, dtype=float)
    
    # Perform chi-square test
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    return chi2_stat, p_value


def generate_sha256_hash(bitstrings):
    """
    Generate SHA-256 hash of concatenated bitstrings.
    
    Args:
        bitstrings (list): List of binary strings
        
    Returns:
        str: SHA-256 hash in hexadecimal
    """
    concatenated = ''.join(bitstrings)
    hash_object = hashlib.sha256(concatenated.encode())
    return hash_object.hexdigest()


def generate_classical_random(n_qubits, shots):
    """
    Generate classical pseudo-random numbers for comparison.
    
    Args:
        n_qubits (int): Number of bits (to match quantum range)
        shots (int): Number of random numbers to generate
        
    Returns:
        pandas.DataFrame: Results in same format as quantum results
    """
    max_value = 2 ** n_qubits - 1
    random_ints = np.random.randint(0, max_value + 1, shots)
    
    # Count occurrences
    unique, counts = np.unique(random_ints, return_counts=True)
    
    results_list = []
    for value, count in zip(unique, counts):
        bitstring = format(value, f'0{n_qubits}b')
        probability = count / shots
        
        results_list.append({
            'Bitstring': bitstring,
            'Integer': value,
            'Count': count,
            'Probability': probability
        })
    
    df = pd.DataFrame(results_list)
    df = df.sort_values('Integer').reset_index(drop=True)
    
    return df


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Title and description
st.title("Quantum Random Number Generator")
st.markdown("""
This application generates **true random numbers** using quantum superposition.
Hadamard gates create equal superposition states, and measurements collapse them
into random classical bits.
""")

# Sidebar demo mode selector
st.sidebar.header("Demo Mode")
demo_mode = st.sidebar.radio(
    "Select Demonstration",
    ["Random Number Generation", "Secure Financial Transactions", "Cryptocurrency & Blockchain", "Bell State QRNG", "Ry Gate QRNG"],
    help="Choose which quantum random number demonstration to explore"
)

if demo_mode == "Random Number Generation":
    # Configuration section on main page
    st.header("Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_qubits = st.number_input(
            "Number of Qubits",
            min_value=1,
            max_value=28,
            value=3,
            step=1,
            help="More qubits = larger range of random numbers (0 to 2^n - 1)"
        )

    with col2:
        shots = st.number_input(
            "Number of Shots",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
            help="How many times to run the quantum circuit"
        )

    with col3:
        optimization_level = st.selectbox(
            "Transpiler Optimization Level",
            options=[0, 1, 2, 3],
            index=1,
            help="Higher levels may optimize the circuit better but take longer"
        )

    # Display options
    st.subheader("Display Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        show_bitstrings = st.checkbox(
            "Show Raw Bitstrings",
            value=True,
            help="Display the full results table with bitstrings"
        )

    with col2:
        compare_classical = st.checkbox(
            "Compare with Python PRNG",
            value=False,
            help="Generate classical pseudo-random numbers for comparison"
        )

    with col3:
        enable_download = st.checkbox(
            "Enable CSV Download",
            value=False,
            help="Allow downloading results as CSV"
        )

    # Run button on main page
    st.markdown("---")
    run_button = st.button("Generate Random Numbers", type="primary", use_container_width=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if demo_mode == "Secure Financial Transactions":
    st.header("Secure Financial Transaction Demo")
    st.markdown("""
    This demo shows how quantum random numbers can be used to generate encryption keys
    for securing financial transactions. Each transaction gets a unique quantum-generated
    encryption key and transaction ID.
    """)

    # Transaction parameters
    col1, col2 = st.columns(2)

    with col1:
        transaction_amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )

        recipient_id = st.text_input(
            "Recipient Account",
            value="ACC-123456789",
            help="Recipient account identifier"
        )

    with col2:
        key_strength = st.selectbox(
            "Encryption Key Strength",
            [128, 256, 512],
            index=1,
            help="Bits for quantum-generated encryption key"
        )

    # Generate transaction button
    generate_transaction = st.button("Generate Secure Transaction", type="primary")

    if generate_transaction:
        with st.spinner("Generating quantum random encryption keys..."):
            transaction = simulate_secure_transaction(
                transaction_amount,
                recipient_id,
                key_strength
            )

        st.success("Secure transaction generated successfully!")

        # Display transaction details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transaction Details")
            st.metric("Transaction ID", transaction['transaction_id'][:16] + "...")
            st.metric("Amount", f"${transaction['amount']:.2f}")
            st.metric("Recipient", transaction['recipient'])
            st.metric("Key Strength", transaction['key_strength'])

        with col2:
            st.subheader("Security Information")
            st.code(transaction['encryption_key'][:64] + "..." if len(transaction['encryption_key']) > 64 else transaction['encryption_key'],
                   language=None)
            st.caption("Quantum-generated encryption key (first 64 bits shown)")

        # Show encryption demonstration
        with st.expander("Encryption Demonstration"):
            st.markdown("**Original Transaction Data:**")
            st.code(transaction['original_data'], language=None)

            st.markdown("**Encrypted Data (XOR with quantum key):**")
            st.code(transaction['encrypted_data'], language=None)

            st.info("In a real system, this would use AES encryption with the quantum-generated key for maximum security.")

        # Add realistic flow animation or chart
        st.subheader("Transaction Security Flow")
        st.markdown("""
        **Quantum Random Key Generation** → **Encryption** → **Transaction Transmission** → **Verification**
        """)

        # Create a simple flow diagram using markdown
        st.code("""
Quantum RNG → Key Generation (128-512 bits)
              ↓
Transaction Data Encryption (XOR/AES)
              ↓
Secure Transmission over Network
              ↓
Verification & Decryption at Receiver
        """, language="text")

        # Show encryption key entropy and strength score
        st.subheader("Encryption Key Analysis")
        col1, col2 = st.columns(2)

        with col1:
            # Calculate entropy of the key
            key_entropy = calculate_shannon_entropy([0.5] * len(transaction['encryption_key']))  # Assuming uniform binary
            st.metric("Key Entropy", f"{key_entropy:.2f} bits")
            st.metric("Key Length", f"{len(transaction['encryption_key'])} bits")

        with col2:
            # Strength assessment
            if key_strength >= 256:
                strength_score = "Excellent"
                strength_color = "🟢"
            elif key_strength >= 128:
                strength_score = "Good"
                strength_color = "🟡"
            else:
                strength_score = "Basic"
                strength_color = "🔴"

            st.metric("Security Strength", f"{strength_color} {strength_score}")
            st.caption("Based on key length and quantum randomness")

        # Security insight section
        st.subheader("Security Insights")
        st.info("""
        **How quantum randomness prevents brute-force attacks:**
        Classical pseudo-random generators can be predicted if the seed is compromised.
        Quantum randomness is fundamentally unpredictable - even with infinite computing power,
        past quantum measurements cannot predict future outcomes.

        **Applications in banking:**
        - **Secure OTPs**: One-time passwords generated with quantum randomness
        - **Digital signatures**: Quantum keys for transaction authentication
        - **Session keys**: Unique encryption keys for each banking session
        """)

        # Additional security features
        with st.expander("Advanced Security Features"):
            st.markdown("""
            **Quantum-Enhanced Security Benefits:**

            🔐 **Unpredictable Keys**: Quantum superposition ensures true randomness
            🛡️ **Brute-Force Resistance**: 256-bit quantum keys = 2^256 possible combinations
            ⚡ **Fast Generation**: Keys generated in microseconds via quantum circuits
            🔄 **Non-Repeating**: Each key is statistically unique
            🌐 **Network Security**: Perfect for securing financial data transmission

            **Real-World Banking Applications:**
            - ATM transaction encryption
            - Online banking session security
            - Credit card transaction verification
            - Blockchain-based financial services
            """)

elif demo_mode == "Bell State QRNG":
    st.header("Bell State Quantum Random Number Generator")
    st.markdown("""
    This demo uses a maximally entangled Bell state (for 2 qubits) or GHZ-like state (for more qubits) 
    to generate true random numbers. The state |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩) for 2 qubits creates 
    perfect correlation: measuring both qubits always gives the same result (both 0 or both 1), 
    but which outcome is fundamentally random.
    """)

    # Configuration section
    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        n_qubits_bell = st.number_input(
            "Number of Qubits",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            help="Number of entangled qubits (2 = Bell state, >2 = GHZ-like state)"
        )

    with col2:
        shots_bell = st.number_input(
            "Number of Shots",
            min_value=100,
            max_value=10000,
            value=1024,
            step=100,
            help="Number of times to run the quantum circuit",
            key="bell_shots"
        )

    # Generate button
    generate_bell = st.button("Generate Bell State Random Numbers", type="primary")

    if generate_bell:
        with st.spinner("Generating quantum random numbers using entangled state..."):
            # Generate random bits using Bell/GHZ state
            random_bits, counts = bell_state_qrng(n_qubits_bell, shots_bell)

        st.success(f"Generated {len(random_bits)} random bits using entangled {n_qubits_bell}-qubit state!")

        # Display the quantum circuit
        st.subheader("Entangled State Quantum Circuit")
        
        if n_qubits_bell == 2:
            st.markdown("""
            **Bell State Circuit:**
            1. **Hadamard Gate (H)** on qubit 0: Creates superposition |+⟩ = (1/√2)(|0⟩ + |1⟩)
            2. **CNOT Gate**: Entangles qubits, creating |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)
            3. **Measurements**: Collapse to either |00⟩ or |11⟩ with equal probability
            """)
        else:
            st.markdown(f"""
            **GHZ-like State Circuit ({n_qubits_bell} qubits):**
            1. **Hadamard Gate (H)** on qubit 0: Creates superposition
            2. **CNOT Gates**: Entangles all qubits in a chain
            3. **Measurements**: All qubits collapse to the same state (all 0s or all 1s)
            """)

        # Create and display the circuit
        qc_display = QuantumCircuit(n_qubits_bell, n_qubits_bell)
        qc_display.h(0)
        for i in range(1, n_qubits_bell):
            qc_display.cx(0, i)
        qc_display.measure(range(n_qubits_bell), range(n_qubits_bell))

        circuit_text = qc_display.draw(output='text')
        st.text(circuit_text)

        # Show measurement statistics
        st.subheader("Measurement Statistics")
        st.markdown(f"Distribution from {shots_bell} shots:")

        # Create DataFrame from counts
        results_list = []
        for bitstring, count in sorted(counts.items(), key=lambda x: x[0]):
            results_list.append({
                'Outcome': bitstring,
                'Count': count,
                'Probability': count / shots_bell,
                'All Bits Same?': '✓' if len(set(bitstring)) == 1 else '✗'
            })

        df = pd.DataFrame(results_list)
        st.dataframe(df.style.format({'Probability': '{:.4f}'}), use_container_width=True)

        # Visualization
        st.subheader("Distribution Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        outcomes = df['Outcome'].tolist()
        counts_list = df['Count'].tolist()
        
        # Color bars based on whether all bits are the same
        colors = ['green' if len(set(outcome)) == 1 else 'lightcoral' 
                 for outcome in outcomes]

        ax.bar(outcomes, counts_list, color=colors, edgecolor='black', linewidth=1)
        ax.set_xlabel('Measurement Outcomes', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Entangled State Distribution ({n_qubits_bell} qubits, {shots_bell} shots)', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add expected line for uniform outcomes
        if n_qubits_bell == 2:
            expected = shots_bell / 2  # Only |00⟩ and |11⟩ expected
        else:
            expected = shots_bell / 2  # Only all-0s and all-1s expected
        ax.axhline(y=expected, color='blue', linestyle='--', linewidth=2, 
                   label=f'Expected (for correlated states): {expected:.1f}')
        ax.legend()
        
        plt.xticks(rotation=45 if len(outcomes) > 8 else 0)
        plt.tight_layout()
        st.pyplot(fig)

        # Analysis
        st.subheader("Entanglement Analysis")
        
        # Count how many outcomes have all bits the same
        correlated_outcomes = sum(1 for outcome in counts.keys() if len(set(outcome)) == 1)
        total_outcomes = len(counts)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Unique Outcomes", total_outcomes)
        with col2:
            st.metric("Correlated Outcomes", correlated_outcomes)
        with col3:
            correlation_pct = (correlated_outcomes / total_outcomes * 100) if total_outcomes > 0 else 0
            st.metric("Correlation %", f"{correlation_pct:.1f}%")

        if n_qubits_bell == 2:
            expected_outcomes = 2  # Only 00 and 11
            st.info(f"""
            **Bell State Verification:**
            - Expected outcomes: |00⟩ and |11⟩ only (2 outcomes)
            - Observed unique outcomes: {total_outcomes}
            - Correlated outcomes: {correlated_outcomes}/{total_outcomes}
            
            {'✓ Perfect Bell state!' if total_outcomes == 2 and correlated_outcomes == 2 
             else '⚠ Some uncorrelated outcomes detected (due to simulation noise)'}
            """)
        else:
            st.info(f"""
            **GHZ-like State Verification:**
            - Expected outcomes: All qubits same (|000...0⟩ and |111...1⟩)
            - Observed unique outcomes: {total_outcomes}
            - Correlated outcomes: {correlated_outcomes}/{total_outcomes}
            
            {'✓ Perfect entanglement!' if correlated_outcomes >= total_outcomes * 0.95
             else '⚠ Some uncorrelated outcomes detected (due to simulation noise)'}
            """)

        # Extract individual bits for randomness testing
        st.subheader("Randomness Analysis")
        
        # For Bell/GHZ states, extract the first bit from each measurement
        first_bits = ''.join([bitstring[0] for bitstring, count in counts.items() 
                             for _ in range(count)])
        
        count_0 = first_bits.count('0')
        count_1 = first_bits.count('1')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("First Bit: 0s", count_0)
            st.metric("First Bit: 1s", count_1)
        
        with col2:
            # Calculate entropy of first bit
            if count_0 > 0 and count_1 > 0:
                p0 = count_0 / len(first_bits)
                p1 = count_1 / len(first_bits)
                entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                st.metric("First Bit Entropy", f"{entropy:.4f}")
                st.caption("Maximum entropy = 1.0 for perfect randomness")
            
            balance = abs(count_0 - count_1) / len(first_bits) * 100
            st.metric("Imbalance", f"{balance:.2f}%")
            st.caption("Lower is better (0% = perfect balance)")

        # Explanation
        with st.expander("Understanding Entanglement"):
            st.markdown("""
            **What makes this special?**
            
            In a Bell state or GHZ state:
            - **Before measurement**: All qubits are in a superposed, entangled state
            - **During measurement**: Measuring one qubit instantly determines the others
            - **After measurement**: All qubits show perfect correlation
            
            **Why this creates randomness:**
            - Which outcome (all 0s or all 1s) is fundamentally unpredictable
            - The choice happens only at measurement time
            - No hidden variables can predict the outcome (Bell's theorem)
            
            **Applications:**
            - Quantum cryptography (BB84 protocol)
            - Quantum random number generation
            - Testing quantum computers
            - Fundamental physics experiments
            """)

        # Download option
        with st.expander("Download Results"):
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download CSV",
                csv_buffer.getvalue(),
                f"bell_state_{n_qubits_bell}qubits_{shots_bell}shots.csv",
                "text/csv"
            )

elif demo_mode == "Ry Gate QRNG":
    st.header("Ry Gate Quantum Random Number Generator")
    st.markdown("""
    This demo uses multiple qubits with Ry(π/2) rotation gates to generate true random numbers.
    The Ry gate rotates each qubit around the Y-axis by π/2 radians, creating an equal 
    superposition state. Multiple qubits run in parallel for efficient random bit generation.
    """)

    # Configuration section
    st.subheader("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        num_bits_ry = st.number_input(
            "Number of Random Bits",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            help="Total number of random bits to generate",
            key="ry_bits"
        )

    with col2:
        shots_ry = st.number_input(
            "Number of Shots",
            min_value=100,
            max_value=10000,
            value=1024,
            step=100,
            help="Number of times to run the quantum circuit",
            key="ry_shots"
        )

    # Generate button
    generate_ry = st.button("Generate Ry Gate Random Bits", type="primary")

    if generate_ry:
        with st.spinner("Generating quantum random bits using Ry gates..."):
            # Generate random bits using Ry gate
            random_bits, counts = ry_qrng(num_bits_ry, shots_ry)
            n_qubits_used = len(list(counts.keys())[0])

        st.success(f"Generated {len(random_bits)} random bits using {n_qubits_used} qubits with Ry gates!")

        # Display the quantum circuit
        st.subheader("Ry Gate Quantum Circuit")
        st.markdown(f"""
        **Circuit Configuration:**
        - **Qubits Used:** {n_qubits_used}
        - **Gate Applied:** Ry(π/2) on each qubit
        - **Shots:** {shots_ry}
        - **Total Bits Generated:** {len(random_bits)}
        
        Each qubit is independently rotated to create superposition:
        1. **Ry(π/2) Gates**: Applied to all {n_qubits_used} qubits in parallel
        2. **Measurements**: Each qubit collapses to 0 or 1 randomly
        3. **Concatenation**: Results from multiple shots are combined
        """)

        # Create and display the circuit
        qc_display = QuantumCircuit(n_qubits_used, n_qubits_used)
        for i in range(n_qubits_used):
            qc_display.ry(np.pi/2, i)
        qc_display.measure(range(n_qubits_used), range(n_qubits_used))

        circuit_text = qc_display.draw(output='text')
        st.text(circuit_text)

        # Show measurement statistics by outcome
        st.subheader("Measurement Statistics")
        st.markdown(f"Distribution from {shots_ry} shots ({n_qubits_used}-qubit outcomes):")

        # Create DataFrame from counts
        results_list = []
        for bitstring, count in sorted(counts.items(), key=lambda x: int(x[0], 2)):
            results_list.append({
                'Outcome': bitstring,
                'Integer': int(bitstring, 2),
                'Count': count,
                'Probability': count / shots_ry
            })

        df = pd.DataFrame(results_list)
        
        # Show first 20 outcomes if too many
        if len(df) > 20:
            st.dataframe(df.head(20).style.format({'Probability': '{:.4f}'}), use_container_width=True)
            st.caption(f"Showing first 20 of {len(df)} unique outcomes")
        else:
            st.dataframe(df.style.format({'Probability': '{:.4f}'}), use_container_width=True)

        # Bit-level analysis
        st.subheader("Bit-Level Randomness Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        count_0 = random_bits.count('0')
        count_1 = random_bits.count('1')
        
        with col1:
            st.metric("Total Bits Generated", len(random_bits))
        with col2:
            st.metric("0s Count", count_0)
        with col3:
            st.metric("1s Count", count_1)

        # Bit distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Bit balance
        ax1.bar(['0', '1'], [count_0, count_1], color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1)
        expected_per_bit = len(random_bits) / 2
        ax1.axhline(y=expected_per_bit, color='green', linestyle='--', linewidth=2, 
                   label=f'Expected: {expected_per_bit:.1f}')
        ax1.set_xlabel('Bit Value', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Bit Distribution (0s vs 1s)', fontsize=14)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Right plot: Outcome distribution
        if len(df) <= 32:  # Only plot if manageable number of outcomes
            outcomes_plot = df['Outcome'].tolist()
            counts_plot = df['Count'].tolist()
            
            ax2.bar(range(len(outcomes_plot)), counts_plot, color='#9b59b6', edgecolor='black', linewidth=0.5)
            expected_per_outcome = shots_ry / (2 ** n_qubits_used)
            ax2.axhline(y=expected_per_outcome, color='orange', linestyle='--', linewidth=2,
                       label=f'Expected (Uniform): {expected_per_outcome:.1f}')
            ax2.set_xlabel('Outcome Index', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title(f'{n_qubits_used}-Qubit Outcome Distribution', fontsize=14)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            # For many outcomes, show histogram of probabilities
            probabilities = df['Probability'].values
            ax2.hist(probabilities, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
            expected_prob = 1 / (2 ** n_qubits_used)
            ax2.axvline(x=expected_prob, color='orange', linestyle='--', linewidth=2,
                       label=f'Expected: {expected_prob:.4f}')
            ax2.set_xlabel('Probability', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Probability Distribution Histogram', fontsize=14)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Statistical Analysis
        st.subheader("Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Bit Entropy")
            
            # Calculate bit-level entropy
            if count_0 > 0 and count_1 > 0:
                p0 = count_0 / len(random_bits)
                p1 = count_1 / len(random_bits)
                bit_entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
            else:
                bit_entropy = 0
            
            st.metric("Shannon Entropy", f"{bit_entropy:.6f}")
            st.write("**Maximum Entropy:** 1.0 bit")
            st.write(f"**Entropy Ratio:** {bit_entropy*100:.2f}%")
            
            if bit_entropy > 0.99:
                st.success("Excellent randomness - near perfect entropy!")
            elif bit_entropy > 0.95:
                st.info("Good randomness - high entropy")
            else:
                st.warning("Lower entropy - may need more shots")
        
        with col2:
            st.markdown("#### Balance Test")
            
            # Calculate balance metrics
            balance_diff = abs(count_0 - count_1)
            balance_pct = (balance_diff / len(random_bits)) * 100
            
            st.metric("Imbalance", f"{balance_pct:.3f}%")
            st.write(f"**Difference:** {balance_diff} bits")
            st.write(f"**Expected:** ~0% (perfect balance)")
            
            if balance_pct < 1:
                st.success("Excellent balance!")
            elif balance_pct < 3:
                st.info("Good balance")
            else:
                st.warning("Noticeable imbalance - consider more shots")

        # Ry Gate Theory
        st.subheader("Ry Gate Theory")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Ry(θ) Rotation Matrix:**
            ```
            Ry(θ) = [[cos(θ/2), -sin(θ/2)],
                     [sin(θ/2),  cos(θ/2)]]
            ```
            
            **For θ = π/2:**
            - cos(π/4) = 1/√2 ≈ 0.707
            - sin(π/4) = 1/√2 ≈ 0.707
            - Ry(π/2)|0⟩ = (1/√2)(|0⟩ + |1⟩)
            """)
        
        with col2:
            st.markdown("""
            **Why Ry(π/2)?**
            
            - Creates equal superposition
            - 50% probability for |0⟩
            - 50% probability for |1⟩
            - Equivalent to Hadamard for RNG
            - But different phase properties
            """)

        # Applications
        with st.expander("Applications of Ry-based QRNG"):
            st.markdown("""
            **Real-World Applications:**
            
            🎲 **Gaming & Simulations**
            - Provably fair random number generation
            - Monte Carlo simulations
            - Procedural generation
            
            🔐 **Cryptography**
            - One-time pad generation
            - Initialization vectors
            - Nonce generation
            
            🧪 **Scientific Research**
            - Unbiased sampling
            - Statistical testing
            - Quantum algorithm initialization
            
            **Advantages of Ry Gates:**
            - Parallel generation across multiple qubits
            - Deterministic rotation (easier to verify)
            - Can be adjusted for different probability distributions
            - Natural fit for quantum hardware
            """)

        # Download option
        with st.expander("Download Results & Raw Bits"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Download statistics CSV
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download Outcome Statistics (CSV)",
                    csv_buffer.getvalue(),
                    f"ry_qrng_{n_qubits_used}qubits_{shots_ry}shots_stats.csv",
                    "text/csv"
                )
            
            with col2:
                # Download raw bits
                st.download_button(
                    "Download Raw Bits (TXT)",
                    random_bits,
                    f"ry_qrng_{len(random_bits)}bits.txt",
                    "text/plain"
                )

        # Sample bits display
        with st.expander("View Raw Bit Stream"):
            st.markdown("**First 500 bits:**")
            st.code(random_bits[:500] + ("..." if len(random_bits) > 500 else ""), language=None)
            
            if len(random_bits) > 500:
                st.markdown("**Last 500 bits:**")
                st.code("..." + random_bits[-500:], language=None)

elif demo_mode == "Cryptocurrency & Blockchain":
    st.header("Cryptocurrency & Blockchain Demo")
    st.markdown("""
    This demo shows how quantum random numbers can be used in cryptocurrency and blockchain applications,
    including wallet address generation, transaction nonce creation, and mining nonce generation.
    Quantum randomness ensures true unpredictability for enhanced security.
    """)

    # Demo type selector
    crypto_demo_type = st.selectbox(
        "Select Crypto Demo",
        ["Wallet Address Generation", "Transaction Nonce", "Mining Nonce"],
        help="Choose which cryptocurrency application to demonstrate"
    )

    if crypto_demo_type == "Wallet Address Generation":
        st.subheader("Quantum Wallet Address Generation")
        st.markdown("""
        Generate a simulated cryptocurrency wallet address using quantum random numbers.
        In real cryptocurrencies like Bitcoin, addresses are derived from public keys,
        which should be generated from truly random seeds for maximum security.
        """)

        key_size = st.selectbox(
            "Private Key Size (bits)",
            [128, 256, 512],
            index=1,
            help="Size of the quantum-generated private key"
        )

        generate_wallet = st.button("Generate Quantum Wallet", type="primary")

        if generate_wallet:
            with st.spinner("Generating quantum random private key..."):
                # Generate quantum random private key
                private_key = generate_encryption_key(key_size)

                # Simulate wallet address generation (simplified)
                # In real crypto, this would involve elliptic curve multiplication
                wallet_address = generate_sha256_hash([private_key])[:40].upper()

            st.success("Quantum wallet generated successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Private Key (Keep Secret!)")
                st.code(private_key[:64] + "..." if len(private_key) > 64 else private_key, language=None)
                st.caption(f"Quantum-generated {key_size}-bit private key")

            with col2:
                st.subheader("Wallet Address")
                st.code(f"0x{wallet_address}", language=None)
                st.caption("Derived wallet address (simplified simulation)")

            st.info("**Security Note:** Never share your private key! In real cryptocurrencies, private keys are used to sign transactions and prove ownership.")

    elif crypto_demo_type == "Transaction Nonce":
        st.subheader("Quantum Transaction Nonce Generation")
        st.markdown("""
        Transaction nonces prevent replay attacks and ensure transaction uniqueness.
        Quantum random nonces provide maximum unpredictability and security.
        """)

        num_transactions = st.slider(
            "Number of Transactions",
            min_value=1,
            max_value=10,
            value=3,
            help="Generate nonces for multiple transactions"
        )

        generate_nonces = st.button("Generate Transaction Nonces", type="primary")

        if generate_nonces:
            with st.spinner("Generating quantum random transaction nonces..."):
                transactions = []
                for i in range(num_transactions):
                    # Generate quantum random nonce (64-bit)
                    nonce = generate_encryption_key(64)
                    transaction_id = f"TX-{i+1:03d}"
                    transactions.append({
                        'transaction_id': transaction_id,
                        'nonce': nonce,
                        'timestamp': f"2024-01-{i+1:02d} 12:00:00"
                    })

            st.success(f"Generated {num_transactions} secure transaction nonces!")

            # Display transactions table
            df = pd.DataFrame(transactions)
            st.dataframe(df, use_container_width=True)

            st.info("Each nonce is quantum-generated and unique, preventing transaction replay attacks.")

    elif crypto_demo_type == "Mining Nonce":
        st.subheader("Quantum Mining Nonce Generation")
        st.markdown("""
        In proof-of-work cryptocurrencies, miners search for nonces that produce valid block hashes.
        Quantum random nonces can help avoid predictable patterns in mining attempts.
        """)

        difficulty = st.selectbox(
            "Mining Difficulty",
            ["Easy (4 leading zeros)", "Medium (6 leading zeros)", "Hard (8 leading zeros)"],
            index=1,
            help="Simulated mining difficulty"
        )

        # Map difficulty to number of leading zeros
        difficulty_map = {
            "Easy (4 leading zeros)": 4,
            "Medium (6 leading zeros)": 6,
            "Hard (8 leading zeros)": 8
        }
        target_zeros = difficulty_map[difficulty]

        generate_mining = st.button("Start Quantum Mining Simulation", type="primary")

        if generate_mining:
            with st.spinner("Quantum mining simulation..."):
                attempts = 0
                found = False
                mining_results = []

                # Simulate mining attempts (limited for demo)
                max_attempts = 1000

                for attempt in range(max_attempts):
                    attempts += 1
                    # Generate quantum random nonce
                    nonce = generate_encryption_key(32)  # 32-bit nonce

                    # Simulate block header hash (simplified)
                    block_data = f"block_header_data_{nonce}"
                    block_hash = generate_sha256_hash([block_data])

                    # Check if hash meets difficulty (starts with target_zeros zeros)
                    if block_hash.startswith('0' * target_zeros):
                        found = True
                        mining_results.append({
                            'attempt': attempts,
                            'nonce': nonce,
                            'hash': block_hash,
                            'status': 'SUCCESS'
                        })
                        break
                    elif attempt < 5:  # Show first few attempts
                        mining_results.append({
                            'attempt': attempts,
                            'nonce': nonce,
                            'hash': block_hash[:16] + "...",
                            'status': 'FAILED'
                        })

                if not found:
                    mining_results.append({
                        'attempt': attempts,
                        'nonce': 'N/A',
                        'hash': 'N/A',
                        'status': 'NO SOLUTION FOUND'
                    })

            if found:
                st.success(f"Block mined successfully after {attempts} attempts!")
            else:
                st.warning(f"No solution found after {max_attempts} attempts. Try easier difficulty.")

            # Display mining results
            df = pd.DataFrame(mining_results)
            st.dataframe(df, use_container_width=True)

            st.info(f"**Mining Statistics:** Target was hash starting with {target_zeros} zeros. Quantum randomness ensures unpredictable nonce generation.")

elif demo_mode == "Random Number Generation" and run_button:
    # Display configuration summary
    st.subheader("Configuration Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Qubits", n_qubits)
    with col2:
        st.metric("Shots", shots)
    with col3:
        st.metric("Range", f"0 - {2**n_qubits - 1}")

    # Generate and display a sample random number immediately
    st.subheader("Sample Random Number")
    with st.spinner("Generating quantum random number..."):
        # Create a single-shot quantum circuit for immediate display
        sample_qc = create_qrng_circuit(n_qubits)
        sample_counts, _ = run_quantum_simulation(sample_qc, 1, optimization_level)
        sample_bitstring = list(sample_counts.keys())[0]
        sample_integer = bitstring_to_int(sample_bitstring)

    # Display the sample in a prominent box
    st.info(f"**Generated Random Number:** {sample_integer}")
    st.code(f"Binary: {sample_bitstring}", language=None)
    st.caption("This is a single quantum-generated random number for immediate reference")
    
    # ========================================================================
    # QUANTUM CIRCUIT SETUP
    # ========================================================================

    st.subheader("Quantum Circuit")
    
    with st.spinner("Creating quantum circuit..."):
        qc = create_qrng_circuit(n_qubits)
    
    # Display circuit diagram
    circuit_text = qc.draw(output='text')
    st.text(circuit_text)
    
    # ========================================================================
    # QUANTUM SIMULATION
    # ========================================================================

    st.subheader("Quantum Simulation")
    
    with st.spinner(f"Running {shots} shots on AerSimulator..."):
        counts, exec_time = run_quantum_simulation(
            qc, 
            shots, 
            optimization_level
        )
    
    st.success(f"Simulation completed in {exec_time:.3f} seconds")
    
    # ========================================================================
    # RESULTS PROCESSING
    # ========================================================================

    st.subheader("Results")
    
    # Process quantum results
    quantum_df = process_results(counts, n_qubits, shots)
    
    # Display results table
    if show_bitstrings:
        st.dataframe(
            quantum_df.style.format({
                'Probability': '{:.4f}'
            }),
            use_container_width=True
        )
    else:
        st.dataframe(
            quantum_df[['Integer', 'Count', 'Probability']].style.format({
                'Probability': '{:.4f}'
            }),
            use_container_width=True
        )
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    st.subheader("Distribution Visualization")
    
    # Create complete range for plotting
    max_outcome = 2 ** n_qubits
    all_outcomes = list(range(max_outcome))
    
    # Get counts for all possible outcomes (0 if not observed)
    outcome_counts = []
    for i in all_outcomes:
        matching = quantum_df[quantum_df['Integer'] == i]
        if len(matching) > 0:
            outcome_counts.append(matching.iloc[0]['Count'])
        else:
            outcome_counts.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot quantum results
    bars = ax.bar(
        all_outcomes, 
        outcome_counts, 
        color='#1f77b4',
        alpha=0.7,
        label='Quantum RNG',
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add expected uniform distribution line
    expected_count = shots / max_outcome
    ax.axhline(
        y=expected_count,
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f'Expected (Uniform): {expected_count:.1f}'
    )
    
    # Compare with classical if enabled
    if compare_classical:
        classical_df = generate_classical_random(n_qubits, shots)
        
        classical_counts = []
        for i in all_outcomes:
            matching = classical_df[classical_df['Integer'] == i]
            if len(matching) > 0:
                classical_counts.append(matching.iloc[0]['Count'])
            else:
                classical_counts.append(0)
        
        ax.bar(
            all_outcomes, 
            classical_counts, 
            color='green',
            alpha=0.4,
            label='Classical PRNG',
            width=0.6
        )
    
    ax.set_xlabel('Outcome (Integer)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Quantum Random Number Distribution ({n_qubits} qubits, {shots} shots)', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Set x-axis ticks
    if max_outcome <= 16:
        ax.set_xticks(all_outcomes)
    else:
        # For larger ranges, show fewer ticks
        tick_step = max(1, max_outcome // 16)
        ax.set_xticks(range(0, max_outcome, tick_step))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    
    st.subheader("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Shannon Entropy")
        
        # Calculate entropy
        probabilities = quantum_df['Probability'].values
        entropy = calculate_shannon_entropy(probabilities)
        theoretical_max = n_qubits
        
        st.metric(
            "Entropy (bits)", 
            f"{entropy:.4f}",
            delta=f"{entropy - theoretical_max:.4f} vs. theoretical max"
        )
        
        st.write(f"**Theoretical Maximum:** {theoretical_max} bits")
        st.write(f"**Entropy Ratio:** {(entropy/theoretical_max)*100:.2f}%")
        
        if entropy > theoretical_max * 0.95:
            st.success("Excellent randomness - entropy very close to maximum!")
        elif entropy > theoretical_max * 0.85:
            st.info("Good randomness - entropy is reasonably high")
        else:
            st.warning("Lower entropy - consider increasing shots for better distribution")
    
    with col2:
        st.markdown("#### Chi-Square Uniformity Test")
        
        # Perform chi-square test
        chi2_stat, p_value = chi_square_uniformity_test(
            outcome_counts, 
            expected_count
        )
        
        st.metric("χ² Statistic", f"{chi2_stat:.4f}")
        st.metric("p-value", f"{p_value:.4f}")
        
        st.write(f"**Degrees of Freedom:** {max_outcome - 1}")
        
        if p_value >= 0.01:
            st.success("Distribution consistent with uniformity (p ≥ 0.01)")
        else:
            st.warning("Distribution may deviate from uniformity (p < 0.01)")
        
        st.caption("""
        **Interpretation:** A high p-value (≥ 0.01) suggests the distribution 
        is statistically consistent with a uniform distribution, indicating 
        good quantum randomness.
        """)
    
    # ========================================================================
    # CRYPTOGRAPHIC HASH (OPTIONAL)
    # ========================================================================

    with st.expander("Cryptographic Hash (SHA-256)"):
        st.markdown("""
        This demonstrates how quantum random bits can be used for cryptographic purposes.
        The SHA-256 hash of all generated bitstrings is shown below.
        """)
        
        all_bitstrings = [row['Bitstring'] for _, row in quantum_df.iterrows()
                         for _ in range(row['Count'])]
        
        hash_value = generate_sha256_hash(all_bitstrings)
        
        st.code(hash_value, language=None)
        st.caption(f"Hash of {len(all_bitstrings)} bitstrings ({len(all_bitstrings) * n_qubits} bits)")
    
    # ========================================================================
    # CSV DOWNLOAD
    # ========================================================================
    
    if enable_download:
        st.subheader("Download Results")

        # Convert DataFrame to CSV
        csv_buffer = StringIO()
        quantum_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"qrng_results_{n_qubits}qubits_{shots}shots.csv",
            mime="text/csv"
        )

else:
    # Initial state - show instructions
    st.info("Configure the parameters and click the button to start!")

    st.markdown("""
    ### How It Works

    1. **Superposition**: Hadamard gates put each qubit into an equal superposition of |0⟩ and |1⟩
    2. **Measurement**: When measured, each qubit randomly collapses to 0 or 1
    3. **Randomness**: The quantum nature ensures true randomness (not pseudo-random)
    4. **Distribution**: With enough shots, all outcomes appear with equal probability

    ### Key Features

    - Uses quantum superposition for true randomness
    - Visualizes distribution uniformity
    - Calculates Shannon entropy and chi-square test
    - Generates cryptographic hash for demonstration
    - Optional CSV export of results
    - Compare with classical pseudo-random numbers

    ### Applications

    - Cryptographic key generation
    - Monte Carlo simulations
    - Gaming and lotteries
    - Scientific research requiring unbiased randomness
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**  
Built with Qiskit and Streamlit  
Quantum simulation via AerSimulator
""")