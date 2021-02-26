#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import numpy as np


# In[3]:


audio_data = 'Amador10-25.wav'
#This returns an audio time series as a numpy array with a default sampling rate(sr) of 22KHZ
sr=22050
x,sr= librosa.load(audio_data,sr=sr)
ipd.Audio(audio_data)


# In[4]:


framesize=1024
hop_length=512
frames = librosa.util.frame(x, frame_length=1024, hop_length=512, axis=0)


# In[5]:


def extract_energy(x,frames):
    energy = np.array([
    sum(abs(x[i:i+framesize]**2))
    for i in range(0, len(x), hop_length)
    ])
    return energy


# In[6]:


eg=extract_energy(x,framesize)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,5))
librosa.display.waveplot(x, sr=sr,alpha=0.4)
plt.figure(figsize=(15,5))
plt.plot(eg, 'r')
plt.xlim(0,len(frames))
plt.xlabel('Frames')
plt.legend(('ENERGY')) 
plt.grid()


# In[7]:


def decision_on_threshold(th,eg):
        x1=[]
        s=[]
        true=[]
        for i in range(len(eg)):   
                if eg[i]>=th:
                        x1.append(1)
                        true.append(i)
                else:
                        x1.append(0)
                s.append(i)              
        return x1,s,true       


# In[30]:


th=2.265
decision,ds,tr=decision_on_threshold(th,eg)
dst=librosa.frames_to_time(ds, sr=sr, hop_length=512, n_fft=None)
c=0
x=0
l=len(decision)
i=0
while i<l :
    j=c
    if decision[j]==0 :
        while decision[j]!=1 and j<l:
            j+=1
            if j==l:
                break
            else:
                c=c+1
        print(" {} {} non speech\n".format(dst[x],dst[c]))
        x=c
    if  decision[j]==1:
        while decision[j]!=0 and j<l:
            j+=1
            if j==l:
                break
            else:
                x+=1
        print("{} {}  speech \n".format(dst[c],dst[x]))
        c=x
        
    

        
    


# In[10]:


from IPython.display import Audio
import soundfile as sf
req=[]
for i in tr:
        j=0
        for j in frames[i]:
                req.append(j)
sf.write('test.wav',req, sr)
Audio(req, rate=sr)


# In[11]:


from pydub import AudioSegment
from pydub.utils import make_chunks

myaudio = AudioSegment.from_file("test.wav" , "wav") 
chunk_length_ms = 2000 
chunks = make_chunks(myaudio, chunk_length_ms)

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print ("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")


# In[12]:


from sklearn.metrics import f1_score
st=[]
et=[]
f=open(r"E:\data\textgrid.txt","rt")
data=f.read()
linepair=data.split()
for i in range(0,len(linepair),2):
    c=i+1
    if i=='xmin' or i=='xmax':
        pass
    else:
        st.append(float(linepair[i]))
        et.append(float(linepair[c]))
start=np.array(st)
end=np.array(et)
startframe=librosa.time_to_frames(start, sr=22050, hop_length=512, n_fft=None)
endframe=librosa.time_to_frames(end, sr=22050, hop_length=512, n_fft=None)
gt=np.zeros(len(eg))
l=len(startframe)
c=0
for i in range(0,l,2):
        j= startframe[i]
        x=endframe[i]
        while j<=x :
                gt[j]=1
                j+=1 
th=2.265
decision,d,f=decision_on_threshold(th,eg)
fscore=f1_score(decision,gt)
print(fscore)

    


# In[ ]:





# In[ ]:





# In[ ]:




