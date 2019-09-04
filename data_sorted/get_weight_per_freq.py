from tqdm import trange
import soundfile as sf
import torch
manifest_r = open('L553_fixedmic_hyper_0.1cm_RT0.2_nongrid_0.5.csv', 'r')

lines = manifest_r.readlines()

# STFT parameter
#n_fft = 8192
n_fft = 1024
win_size = n_fft
hop_length = int(win_size/2)
window_path = '/data2/kenkim/DCUnet/window_' + str(win_size) + '.pth'
window = torch.load(window_path)
stft = lambda x: torch.stft(x, n_fft, hop_length, win_length = win_size, window=window.cpu())

# weights
nFreq = int(n_fft/2+1)
weights = torch.zeros(nFreq)

for i in trange(0, len(lines)):
    line = lines[i]
    line_splited = line.split(',')
    clean_path = line_splited[1].replace('\n', '')
    audio, sr = sf.read(clean_path)

    # STFT
    STFT = stft(torch.FloatTensor(audio)) # FxTx2 (2 = real/imag)
    real, imag = STFT[..., 0], STFT[..., 1]
    mag = torch.sqrt(real*real + imag*imag)
    magsum = mag.sum(1)
    weights += magsum

torch.save(weights, 'weights_per_freq_' + str(n_fft) + '.pth')


manifest_r.close()