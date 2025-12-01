import hashlib

files = [
    'Quantum_entanglement_vs_classical_correlation_video.gif',
    'Quantum_measurement_animation.gif',
    'GWM_HahnEcho.gif'
]

base = 'https://upload.wikimedia.org/wikipedia/commons'

for f in files:
    h = hashlib.md5(f.encode('utf-8')).hexdigest()
    url = f"{base}/{h[0]}/{h[:2]}/{f}"
    print(url)
