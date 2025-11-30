import sys, site, sysconfig, traceback
print('python:', sys.executable)
print('sys.path:')
for p in sys.path:
    print('  ', p)
print('site.getsitepackages():', getattr(site, 'getsitepackages', lambda: 'n/a')())
print('getusersitepackages():', site.getusersitepackages())

try:
    import sounddevice
    print('sounddevice import OK, file:', getattr(sounddevice, '__file__', None))
except Exception:
    print('sounddevice import failed:')
    traceback.print_exc()

try:
    import librosa
    print('librosa import OK, file:', getattr(librosa, '__file__', None))
except Exception:
    print('librosa import failed:')
    traceback.print_exc()
