# FastAudioSR
This repository is organized by importing only 48Khz Audio Resolution code from [HierSpeech++](https://github.com/sh-lee-prml/HierSpeechpp).

# Usage
```python
from FastAudioSR import FASR
fasr = FASR('FastAudioSR/SR48k.pth')
fasr.run('ref_16k.wav', 'ref_48k.wav')
```