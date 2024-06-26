import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import webvtt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Processor, HubertModel
from resemblyzer import VoiceEncoder, preprocess_wav  # pip install resemblyzer
import librosa

def load_audio(file_path, target_sr=16000):
    signal, sr = librosa.load(file_path, sr=target_sr)
    return signal, sr


def resample_audio(signal, orig_sr, target_sr):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(torch.tensor(signal).float()).numpy()


def extract_embeddings(signal, sr, segment_length=1.0, overlap=0.25):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")
    embeddings = []
    times = []
    start = 0

    step = segment_length - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than segment length")

    while start < len(signal) / sr:
        end = start + segment_length
        segment = signal[int(start * sr):int(end * sr)]
        if len(segment) > 0:
            embedding = classifier.encode_batch(torch.tensor(segment).unsqueeze(0)).squeeze().numpy()
            embeddings.append(embedding)
            times.append((start, min(end, len(signal) / sr)))
        start += segment_length - overlap
    return np.array(embeddings), times


def extract_embeddings_hubert(signal, sr, segment_length, overlap):
    from transformers import Wav2Vec2Processor, HubertModel
    import torch
    import numpy as np

    step = segment_length - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than segment length")

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    step = segment_length - overlap
    embeddings = []
    times = []

    num_segments = int(np.ceil(len(signal) / sr / step))

    for i in range(num_segments):
        start = int(i * step * sr)
        end = start + int(segment_length * sr)
        segment = signal[start:end]

        if len(segment) == 0:
            continue

        inputs = processor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
        times.append((start / sr, end / sr))

    return np.array(embeddings), times


def extract_embeddings_resemblyzer(signal, sr, segment_length, overlap):
    from resemblyzer import VoiceEncoder
    import numpy as np

    encoder = VoiceEncoder()

    step = segment_length - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than segment length")

    step = segment_length - overlap
    embeddings = []
    times = []

    num_segments = int(np.ceil(len(signal) / sr / step))

    for i in range(num_segments):
        start = int(i * step * sr)
        end = start + int(segment_length * sr)
        segment = signal[start:end]

        if len(segment) == 0:
            continue

        embedding = encoder.embed_utterance(segment)
        embeddings.append(embedding)
        times.append((start / sr, end / sr))

    return np.array(embeddings), times


def perform_clustering(embeddings, distance_threshold=1.5, dbscan_eps=0.5, dbscan_min_samples=5):
    clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
    initial_labels = clustering.fit_predict(embeddings)

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
    dbscan_labels = dbscan.fit_predict(embeddings)

    combined_labels = initial_labels.copy()
    noise_label = max(initial_labels) + 1

    for i in range(len(initial_labels)):
        if initial_labels[i] == -1:
            if dbscan_labels[i] != -1:
                combined_labels[i] = dbscan_labels[i]
            else:
                combined_labels[i] = noise_label
                noise_label += 1

    return combined_labels


def time_to_seconds(time_str):
    h, m, s = map(float, time_str.replace(',', '.').split(":"))
    return h * 3600 + m * 60 + s


def load_vtt(filepath):
    captions = []
    for caption in webvtt.read(filepath):
        start = time_to_seconds(caption.start)
        end = time_to_seconds(caption.end)
        captions.append({
            'start': start,
            'end': end,
            'text': caption.text
        })
    return captions


def match_speakers_to_segments(captions, labels, times):
    matched_segments = []
    for i, caption in enumerate(captions):
        start, end, text = caption['start'], caption['end'], caption['text']
        caption_mid = (start + end) / 2

        # Найти ближайший временной сегмент для текущего текста
        # times предполагается как список кортежей (start_time, end_time)
        times_mid = [(t[0] + t[1]) / 2 for t in times]
        closest_idx = np.argmin([abs(caption_mid - t) for t in times_mid])

        speaker = labels[closest_idx]
        matched_segments.append({
            'start': start,
            'end': end,
            'text': text,
            'speaker': speaker
        })

    return matched_segments


def postprocess_segments(matched_segments):

    postprocessed_segments = []
    for segment in matched_segments:
        if not postprocessed_segments:
            postprocessed_segments.append(segment)
        else:
            last_segment = postprocessed_segments[-1]
            if last_segment['speaker'] == segment['speaker'] and last_segment['end'] >= segment['start']:
                # Объединяем сегменты одного говорящего
                last_segment['end'] = max(last_segment['end'], segment['end'])
                last_segment['text'] += " " + segment['text']
            else:
                postprocessed_segments.append(segment)
    return postprocessed_segments


def seconds_to_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace('.', ',')


def format_vtt(segments):
    vtt_output = "WEBVTT\n\n"
    for segment in segments:
        start = seconds_to_time(segment['start'])
        end = seconds_to_time(segment['end'])
        speaker = f"[Speaker_{segment['speaker']}]"
        text = segment['text']
        vtt_output += f"{start} --> {end}\n{speaker}: {text}\n\n"
    return vtt_output


def main(wav_file_path, vtt_file_path, output_vtt_file_path):
    # Загружаем аудиофайл с нужной частотой дискретизации
    signal, sr = load_audio(wav_file_path)

    # Используем HuBERT для извлечения эмбеддингов
    embeddings_hubert, times_hubert = extract_embeddings_hubert(signal, sr, segment_length=5.0, overlap=2.5)
    print(f"Extracted {len(embeddings_hubert)} embeddings from HuBERT.")

    # Используем Resemblyzer для извлечения эмбеддингов
    embeddings_resemblyzer, times_resemblyzer = extract_embeddings_resemblyzer(signal, sr, segment_length=1.0,
                                                                               overlap=0.3)
    print(f"Extracted {len(embeddings_resemblyzer)} embeddings from Resemblyzer.")

    # Приведение эмбеддингов к одной длине
    min_length = min(len(embeddings_hubert), len(embeddings_resemblyzer))
    embeddings_hubert = embeddings_hubert[:min_length]
    embeddings_resemblyzer = embeddings_resemblyzer[:min_length]
    times_hubert = times_hubert[:min_length]  # Синхронизация временных меток с обрезанными эмбеддингами

    # Приведение эмбеддингов к одной размерности (если необходимо)
    if embeddings_hubert.shape[1] != embeddings_resemblyzer.shape[1]:
        embeddings_resemblyzer = np.resize(embeddings_resemblyzer, (min_length, embeddings_hubert.shape[1]))

    # Комбинируем эмбеддинги для кластеризации
    combined_embeddings = np.concatenate((embeddings_hubert, embeddings_resemblyzer), axis=1)
    print(f"Combined {len(combined_embeddings)} embeddings.")

    # Выполнение кластеризации
    labels = perform_clustering(combined_embeddings, distance_threshold=4.5, dbscan_eps=0.8, dbscan_min_samples=2)
    print(f"Generated {len(set(labels))} unique speaker labels.")

    # Загрузка VTT файла с субтитрами
    captions = load_vtt(vtt_file_path)

    # Соответствие говорящих сегментам на основе временной информации
    matched_segments = match_speakers_to_segments(captions, labels, times_hubert)
    if matched_segments is None:
        print("Error: No matched segments found.")
        return

    # Вывод совпавших сегментов
    #for seg in matched_segments:
    #    print(f"Matched Segment: {seg}")

    # Постобработка сегментов для обеспечения согласованности и правильности
    postprocessed_segments = postprocess_segments(matched_segments)

    # Вывод постобработанных сегментов
    #for seg in postprocessed_segments:
    #    print(f"Postprocessed Segment: {seg}")

    # Форматирование сегментов в файл VTT
    vtt_output = format_vtt(postprocessed_segments)

    # Сохранение VTT файла
    with open(output_vtt_file_path, 'w', encoding='utf-8') as f:
        f.write(vtt_output)
    print(f"VTT file saved to {output_vtt_file_path}")


# Example usage
wav_file_path = r"D:\Projects\VideoTransl\dev\dev.mp4.vocals.wav"
vtt_file_path = r"D:\Projects\VideoTransl\dev\16hzconv.vtt"
output_vtt_file_path = r"D:\Projects\VideoTransl\dev\output.vtt"
main(wav_file_path, vtt_file_path, output_vtt_file_path)
