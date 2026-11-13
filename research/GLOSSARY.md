# Glossary

**Amplitude:** Sample value or signal magnitude. In a digital waveform it is represented by discrete numerical samples.

**Attack:** The initial portion of an amplitude envelope. In this project, attack time controls how quickly the kick's amplitude rises.

**Autocorrelation:** A measure of similarity between a signal and delayed versions of itself. Periodicity can be estimated from the lag of a strong autocorrelation peak.

**Band-energy share:** The proportion of measured spectral power found inside a chosen frequency interval. It depends on the selected intervals and analysis method.

**Crest factor:** Ratio of peak amplitude to RMS amplitude, usually expressed in decibels. It describes peakiness relative to average energy.

**dBFS:** Decibels relative to digital full scale. 0 dBFS is the maximum representable sample peak in a normalized digital signal.

**Decay:** The portion of an envelope after its initial rise. In this synthesizer it controls the amplitude fall; it is distinct from pitch decay.

**FFT:** Fast Fourier Transform, an efficient algorithm for computing the discrete Fourier transform.

**Fundamental frequency (F0):** The lowest frequency of a periodic harmonic signal. In a complex or transient signal, estimating it can be ambiguous.

**Harmonic partial:** A sinusoidal component at an integer multiple of a fundamental frequency.

**Phase:** The position of an oscillator within its cycle. Integrating instantaneous frequency produces continuous phase for a pitch-changing oscillator.

**RMS:** Root mean square, a measure of average signal magnitude over a defined interval.

**Saturation:** Nonlinear processing that changes waveform shape and introduces additional spectral components. Here it is implemented with a tanh function.

**Spectral centroid:** Magnitude-weighted mean frequency of a spectrum. It is often used as a spectral-balance descriptor but is not a direct measure of perceived warmth or brightness.

**Spectrogram:** A time-frequency representation showing how spectral energy changes over time.

**STFT:** Short-Time Fourier Transform. It applies Fourier analysis to overlapping windowed segments of a signal to show how the spectrum changes over time.

**Transient:** A brief onset component that contributes to the initial attack definition.

**True peak:** An estimate of the maximum reconstructed waveform level between digital samples. It can exceed the largest stored sample value; this project does not currently calculate it.
