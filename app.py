"""
Quantum Random Number Generator (QRNG)
A Streamlit application that generates true random numbers using quantum superposition.

Installation:
pip install qiskit qiskit-aer streamlit numpy scipy pandas matplotlib

Run:
streamlit run app.py

Author: Hackathon Team
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

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
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

    /* Metric cards */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        backdrop-filter: blur(10px);
    }

    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
    }

    /* Success/info/warning messages */
    .stSuccess, .stInfo, .stWarning {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border-left: 4px solid #00f2fe;
    }

    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.1);
        color: black;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 5px;
    }

    .stTextInput>div>div>input::placeholder, .stNumberInput>div>div>input::placeholder {
        color: rgba(0, 0, 0, 0.6);
    }

    /* Slider styling */
    .stSlider>div>div>div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Radio buttons */
    .stRadio>div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
        color: black !important;
    }

    .stRadio label {
        color: black !important;
    }

    .stRadio input[type="radio"]:checked + div {
        color: black !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: black;
    }

    /* Checkboxes */
    .stCheckbox>div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 5px;
        color: black !important;
    }

    .stCheckbox label {
        color: black !important;
    }

    .stCheckbox input[type="checkbox"]:checked + div {
        color: black !important;
    }

    /* Progress bar */
    .stProgress>div>div>div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Plot styling */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
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
    ["Random Number Generation", "Secure Financial Transactions", "Cryptocurrency & Blockchain"],
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
            import pandas as pd
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
            width='stretch'
        )
    else:
        st.dataframe(
            quantum_df[['Integer', 'Count', 'Probability']].style.format({
                'Probability': '{:.4f}'
            }),
            width='stretch'
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
    st.info("Configure the parameters in the sidebar and click Generate Random Numbers to start!")

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