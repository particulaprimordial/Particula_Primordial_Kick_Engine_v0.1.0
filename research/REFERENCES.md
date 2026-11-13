# References and Further Reading

These resources support the project's signal-processing concepts. They are not evidence that any particular preset is objectively superior.

1. Smith, J. O. (2011). *Spectral Audio Signal Processing*. W3K Publishing. Online text: [DSPRelated — Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/). Useful for Fourier analysis, spectral interpretation, and time-frequency methods.
2. Zölzer, U. (Ed.). (2011). *DAFX: Digital Audio Effects* (2nd ed.). Wiley. [DOI: 10.1002/9781119991298](https://doi.org/10.1002/9781119991298). Background on digital audio processing, nonlinear processing, filters and spectral effects.
3. SciPy Contributors. *ShortTimeFFT — SciPy documentation*. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html). Software reference for computing the short-time Fourier transform and interpreting its time/frequency grid.
4. SciPy Contributors. *Signal processing tutorial: Short-Time Fourier Transform*. [Documentation](https://docs.scipy.org/doc/scipy/tutorial/signal.html#short-time-fourier-transform). Introductory explanation of windowed, overlapping Fourier analysis.
5. International Telecommunication Union. (2023). *Recommendation ITU-R BS.1770-5: Algorithms to measure audio programme loudness and true-peak audio level*. [Recommendation](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en). Included as a metering reference. This project does not implement BS.1770 and must not claim LUFS or ITU loudness compliance.

## Recommended reading order

1. Smith, chapters on Fourier transforms and the STFT.
2. SciPy's STFT tutorial and `ShortTimeFFT` API documentation.
3. Zölzer, chapters on nonlinear processing and spectral processing.
4. ITU-R BS.1770-5 only when extending the project toward programme loudness and true-peak metering.
