"""
IBM Quantum Setup Template

Copy this file to setup_ibm.py and add your credentials.
DO NOT commit setup_ibm.py with your actual credentials!
"""

from src.hardware_deployment import IBMQuantumDeployer

# TODO: Replace with your actual IBM Quantum credentials
# Get these from: https://quantum.ibm.com/account

YOUR_TOKEN = "your_api_token_here"  # 44-character API token from IBM Quantum Platform
YOUR_CRN = "your_crn_instance_here"  # CRN from your IBM Quantum instance

# Setup IBM Quantum connection
deployer = IBMQuantumDeployer()
deployer.setup_ibm_account(
    token=YOUR_TOKEN,
    instance=YOUR_CRN
)

print("✅ IBM Quantum configured successfully!")
print("\nTo test your connection, run:")
print("  python test_backends.py")
