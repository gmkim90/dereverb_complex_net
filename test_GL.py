import librosa
import numpy as np
import pickle as pkl


def linspec(x, win_length, hop_length, n_fft):
    min_level_db = -100
    ref_level_db = 20
    X = librosa.stft(x, n_fft=n_fft, win_length=win_length, window='hann', hop_length=hop_length)
    Xmag_unnormal = np.abs(X)
    Xphs = np.angle(X)
    Xmag = 20 * np.log10(np.maximum(1e-5, Xmag_unnormal)) - ref_level_db
    Xmag = np.clip(-(Xmag - min_level_db) / min_level_db, 0, 1)
    return Xmag.T, Xphs.T, Xmag_unnormal.T

def spectrogram2wav(spectrogram, n_fft, win_length, hop_length, num_iters):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    min_level_db = -100
    ref_level_db = 20

    spec = spectrogram.T
    # denormalize
    spec = (np.clip(spec, 0, 1) * - min_level_db) + min_level_db
    spec = spec + ref_level_db

    # Convert back to linear
    spec = np.power(10.0, spec * 0.05)

    return _griffin_lim(spec ** 1.5, n_fft, win_length, hop_length, num_iters)  # Reconstruct phase


def _griffin_lim(S, n_fft, win_length, hop_length, num_iters):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    #angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(num_iters):
        #if i > 0:
        #    angles = np.exp(1j * np.angle(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    return y


def ISTFT_ground_truth_phase(mag, phs, win_length, hop_length):
    y = librosa.istft(mag.astype(np.complex).T * phs.T, hop_length=hop_length, win_length=win_length)

    return y

if __name__=='__main__':
    sample_rate = 16000
    n_ffts      = [1024]#[1024, 2048, 4096, 8192, 8192]
    win_lengths = [800]#[800 , 1600, 3200, 6400, 8192]
    hop_lengths = [200]#[200 ,  400,  800, 1600, 4096]
    for f, w, h in zip(n_ffts, win_lengths, hop_lengths): 
        print(f, w, h)
        n_fft = f #2048
        win_length = w #int(np.ceil(100 * sample_rate / 1000))
        hop_length = h #int(np.ceil(25 * sample_rate / 1000))

        x, _ = librosa.load('clean_s0.wav', sr=sample_rate)
        Xmag, Xphs, Xmag_unnormal = linspec(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        wav = spectrogram2wav(Xmag, n_fft=n_fft, win_length=win_length, hop_length=hop_length, num_iters=1)
        wav_gt = ISTFT_ground_truth_phase(Xmag_unnormal, Xphs, win_length=win_length, hop_length=hop_length)
        librosa.output.write_wav('recon_f{}_w{}_h{}_i1.wav'.format(f, w, h), wav, 16000)
        librosa.output.write_wav('GT_ISTFT_f{}_w{}_h{}.wav'.format(f, w, h), wav_gt, 16000)
