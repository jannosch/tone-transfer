# tone-transfer

Ein Projekt von Moritz Blum (JLU) und Jannis Müller (THM).
Basierend auf den Arbeiten:
- [DDSP: Differentiable Digital Signal Processing](https://doi.org/10.48550/arXiv.2001.04643)
- [Tone Transfer](https://ceur-ws.org/Vol-2903/IUI21WS-HAIGEN-3.pdf)
- [SPICE](https://doi.org/10.1109/TASLP.2020.2982285)
- [WaveNet](https://doi.org/10.48550/arXiv.1609.03499)

Mit "tone-transfer" lässt sich der Klancharakter eines Instruments auf jede belibige monophone Audioquelle übertragen. Anbei sind beispielsweise zwei eigene Trompetenaufnahmen zum Testen. Das Ziel ist es, die aufgenommene Sequenz so klingen zu lassen, als würde sie von dem trainierten Instrument (in unserem Fall eine E-Gitarre) gespielt werden.

## E-Gitarre
Die Netze können mit jedem monophonen Instrument trainiert werden. Wir haben uns für eine E-Gitarre der Marke "Duesenberg" entschieden, die gerade für >2.500€ erhältlich sind. In der Praxis könnte das Netz bei der Musikproduktion für's Prototyping genutzt werden, um zu Testen, ob die E-Gitarre in einen Song passen würde. Im Live-Einsatz könnte mit einer Implementierung als Audio-Plugin z.B. [hier](https://magenta.tensorflow.org/ddsp-vst) benutzt werden, falls das Instrument benötigt wird, aber nicht vorhanden ist.

# Inhalt
- `tone_transfer_train.ipynb`: Trainingsnotebook mit Datenvorbereitung
- `tone_transfer_inference.ipynb`: Testnotebook mit Möglichkeit zum Upload von eigenen Audiodateien
- `model.zip`: Das Modell. Kann bei der Inferenz hochgeladen werden.
- `tequila.wav` & `mas_que_nada.wav`: Trompetenaufnahmen für das Testen

## Datensatz
Der Datensatz befindet sich in [diesem Google-Drive Ordner](https://drive.google.com/drive/folders/1Y2HU3L9bbDXopPPFhmfxj9h36lKI5bkP?usp=sharing). Eine zwanzigminütige Aufnahme der E-Gitarre wurde in drei Ordner aufgeteilt: `train` (~70%), `val`(~20%) und `test`(~10%). 

Die Audiodateien wurden für die drei Phasen in jeweils 4 Sekunden lange Abschnitte unterteilt. Mithilfe des bestehenden Netzes "SPICE" wurde die Tonhöhe analysiert und zusammen mit der Lautstärke als [TFRecord](https://github.com/google/tensorflow-recorder) abgespeichert. Da dieser Prozess mehr Rechenaufwand als das Training selbst benötigt, wurde dies im Vorfeld berechnet und abgespeichert. 

## Ausführung
Die Jupyter-Notebooks sind für die Ausführung in Google Colab konzipiert. Eine genaue Anleitung zur Ausführung befindet sich in den jeweiligen Notebooks.

# Erläuterung der Grundidee
Auszug aus dem Paper:

Viele frühere Ansätze versuchen, Audio in Frequenz oder Zeitbereich zu generieren. Damit ist die Menge der erzeugbaren Klänge enorm, wobei in der Regel nur ein ganz bestimmter Klang erwartet wird. Dazu kommt, dass das Netz die komplexe Kodierung von Audiosignalen lernen muss. Beispiele dafür sind die Zusammenhänge zwischen Samples und Frequenz im Zeitbereich und von Grundfrequenz zu Oberschwingungen im Frequenzbereich. Dadurch geht ein großteil der Trainingskapazitäten des Netzes für das lernen der Kodierungen verloren. Für komplexe Signale, wie Sprache, ist dies zwar notwendig, aber bei Instrumenten handelt es sich meist um deutlich einfachere Klänge, nämlich um angeregte harmonische Schwingungen. Dieses Domänenwissen nutzt das Paper “Differential Signal Processing” (DDSP) aus. Es stellt eine Architektur vor, indem das Netz das Audiosignal nicht direkt erzeugt, sondern nur die Parameter von zwei verschiedenen Synthesizern steuert. Der erste Synthesizer, ein Additiver Synthesizer, dient der Erzeugung von Grund- und Obertönen der melodischen Klanganteile. Der zweite Synthesizer, ein Noise-Synthesizer, ist für die nicht-melodischen Klanganteile zuständig, wie das Anschlagen einer Saite oder das Blasen in eine Trompete.

