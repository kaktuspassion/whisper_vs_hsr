# README.md

## 1. Project Proposal 

Mandarin is a tonal language and therefore tone plays an important role in the distinction of characteristics and understanding of speech. Absence of tones in whispered speech may negatively affect speech recognition. This paper will compare the performance of the cutting-edge Automatic Speech Recognition (ASR) system, Whisper, to manual speech recognition when it comes to whispered Mandarin speeches. The largest model of Whisper will be utilized here, representing the most advanced and accurate recognition results of that model. As planned, I will collect 30 recordings of whispered Chinese from a male and a female speaker and then assign them to Whisper for recognition and 5 participants for transcription task respectively. The results between the two will be compared at the end. Word Error Rate (WER) will be chosen as the criterion for recognition performance. 



## 2. Files 

`Recordings` folder contains all audio materials for the transcription task. 

`Answer_sheet.docx` is a form for participants to fill in the transcription content.

`Transcriptions.txt` and `Transcription.xlsx` stores all the transcripts from the ASR system and from all 8 participants, retaining the format in which they first reached the researcher. 

`Transcriptions.csv` stores all the transcripts as above. This is basics for future CER calculation, so all non-characters are removed (e.g. punctuations and blank spaces). 

`CER.py` is a python script to iterate all transcription groups (WHISPER, Participant 1 to Participant 8) and compare them with the Gold answer. Through this program all CER were calculated and all charts and figures are generated.  

`CER_Results.csv` stores CERs calculated from WHISPER, 8 participants and mean of participants. 

`whisper_large_v3.ipynb` is ran on colab to get WHISPER transcription results. 
