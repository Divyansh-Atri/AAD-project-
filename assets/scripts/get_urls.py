import hashlib

files = [
    'Animated_Rotating_Frame.gif',
    'Wavepacket1.gif',
    'Quantum_measurement_of_single_photon_polarization.gif'
]

base = 'https://upload.wikimedia.org/wikipedia/commons'

for f in files:
    # Wikimedia uses md5 of the filename (spaces replaced by underscores, which they already are here)
    # to determine the directory structure.
    # However, sometimes it's not exact. But usually it is.
    # Let's try.
    h = hashlib.md5(f.encode('utf-8')).hexdigest()
    url = f"{base}/{h[0]}/{h[:2]}/{f}"
    print(url)
