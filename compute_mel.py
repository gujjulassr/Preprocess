import sys,os
import numpy as np
import torch
from scipy.io import wavfile
from scipy.io.wavfile import read
from tqdm import tqdm
import librosa



sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT







fft_length=1024
frame_step=256 
frame_length=1024
mels=80


MAX_WAV_VALUE = 32768.0
stft = TacotronSTFT(filter_length=fft_length,
                                 hop_length=frame_step,
                                 win_length=frame_length,
                                 sampling_rate=22050,
                                 mel_fmin=0.0, mel_fmax=8000.0)


def kaldi_pad(wav, frame_shift):
    old_length=len(wav)
    num_frames = (old_length + frame_shift // 2) // frame_shift
    new_length = num_frames*frame_shift
    diff = new_length-old_length

    if diff > 0:
        wav = np.concatenate([wav, np.zeros(diff)])
    else:
        wav = wav[:new_length]

    return wav

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return  data, sampling_rate      #torch.from_numpy(data).float(), sampling_rate

def get_spec_mel(audio):
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec, spec,excep = stft.mel_spectrogram(audio_norm)
    if not excep:
        melspec = torch.squeeze(melspec, 0)
        spec = torch.squeeze(spec, 0)
        return melspec, spec
    else:
        return None,None





input_dir='/home/samara/GameChanger/3speakers_data/female_telugu_english'
output_dir='/home/samara/GameChanger/3speakers_data/female_telugu_english_mels'

os.makedirs(output_dir,exist_ok=True)

for file_name in tqdm(os.listdir(input_dir)):

    if file_name.endswith('.wav'):

        audio_path=os.path.join(input_dir,file_name)


        audio, sr = load_wav_to_torch(audio_path)
        audio = audio.astype(np.float32)

        if sr!=22050:
            audio=librosa.resample(audio,orig_sr=sr,target_sr=22050)

        audio = kaldi_pad(audio, 256)
        melspectrogram, spectrogram = get_spec_mel(torch.from_numpy(audio).float())
        if melspectrogram is None:
            with open('errored_files.txt','a') as f:
                f.write(audio_path+'\n')
            continue
        melspectrogram=melspectrogram.T
        melspectrogram=(np.log10(np.exp(melspectrogram))+5)/5

        base_name=os.path.basename(audio_path).strip().split('.wav')[0]


        torch.save(melspectrogram,f'{output_dir}/{base_name}.pt')

        