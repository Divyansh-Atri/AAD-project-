import hashlib

files = [
    'Quantum_circuit_simulations_of_3-qubit_QFT.gif',
    'Quantum_Tunnelling_animation.gif'
]

base = 'https://upload.wikimedia.org/wikipedia/commons'

for f in files:
    h = hashlib.md5(f.encode('utf-8')).hexdigest()
    url = f"{base}/{h[0]}/{h[:2]}/{f}"
    print(url)
