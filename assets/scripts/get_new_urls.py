import hashlib

files = [
    'Animation_of_quantum_mechanics.gif',
    'Quantum_circuit_simulations_of_3-qubit_QFT.gif'
]

base = 'https://upload.wikimedia.org/wikipedia/commons'

for f in files:
    h = hashlib.md5(f.encode('utf-8')).hexdigest()
    url = f"{base}/{h[0]}/{h[:2]}/{f}"
    print(url)
