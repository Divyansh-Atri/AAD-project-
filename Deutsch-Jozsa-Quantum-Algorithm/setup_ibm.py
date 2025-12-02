from src.hardware_deployment import IBMQuantumDeployer

# Your IBM Quantum credentials
YOUR_TOKEN = "aa_X0hnufY-2Gl66v7FA4nyXME45uPq2-hGGKFy4_nEJ"
YOUR_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/46cb9e194228436da35be525af94b959:d8c0f135-0ab6-4d08-a16d-3139ff75c392::"

deployer = IBMQuantumDeployer()
deployer.setup_ibm_account(
    token=YOUR_TOKEN,
    instance=YOUR_CRN  # Using your CRN for open-instance
)

print("✅ IBM Quantum configured successfully!")