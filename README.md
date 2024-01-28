# tone-transfer

Ein Projekt von Moritz Blum (JLU) und Jannis Müller (THM).
Basierend auf den Arbeiten:
- [DDSP: Differentiable Digital Signal Processing](https://doi.org/10.48550/arXiv.2001.04643)
- [Tone Transfer](https://ceur-ws.org/Vol-2903/IUI21WS-HAIGEN-3.pdf)
- [WaveNet](https://doi.org/10.48550/arXiv.1609.03499)

Mit "tone-transfer" lässt sich der Klancharakter eines Instruments auf jede belibige monophone Audioquelle übertragen. Das kann die aufgenommene Stimme, ein anderes Instrument, Naturgeräusche usw. sein. Das Ziel ist es, die aufgenommene Sequenz so klingen zu lassen, als würde sie von dem trainierten Instrument (in unserem Fall eine E-Gitarre) gespielt werden.

## E-Gitarre
Die Netze können mit jedem monophonen Instrument trainiert werden. Wir haben uns für eine E-Gitarre der Marke "Duesenberg" entschieden, die gerade für >2.500€ erhältlich sind. In der Praxis könnte das Netz bei der Musikproduktion für's Prototyping genutzt werden, um zu Testen, ob die E-Gitarre in einen Song passen würde. Im Live-Einsatz könnte mit einer Implementierung als Audio-Plugin z.B. [hier](https://magenta.tensorflow.org/ddsp-vst) benutzt werden, falls das Instrument benötigt wird, aber nicht vorhanden ist.

# Inhalt
- `tone_transfer_train.ipynb`: Trainingsnotebook mit Datenvorbereitung
- `tone_transfer_inference.ipynb`: Testnotebook mit Upload von eigenen Audiodateien
- `model.zip`: Das Modell. Kann bei der Inferenz hochgeladen werden.
- `tequila.wav` & `mas_que_nada.wav`: Trompetenaufnahmen für das Testen
# TODO: Datensatz

## Ausführung
Die Jupyter-Notebooks sind für die Ausführung in Google Colab konzipiert. Eine genaue Anleitung zur Ausführung befindet sich in den jeweiligen Notebooks.
