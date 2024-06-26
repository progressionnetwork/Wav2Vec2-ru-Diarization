from typing import Optional

import torch
import torchaudio
from matplotlib import pyplot as plt
from pyannote.pipeline.blocks import HierarchicalAgglomerativeClustering
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from spectralcluster import SpectralClusterer
from speechbrain.pretrained import EncoderClassifier
import webvtt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering, KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Processor, HubertModel
from resemblyzer import VoiceEncoder, preprocess_wav  # pip install resemblyzer
import librosa
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from transformers import Wav2Vec2Processor, HubertModel
from resemblyzer import VoiceEncoder
import concurrent.futures
from functools import partial, lru_cache
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from kneed import KneeLocator
import torch.nn as nn
from torchaudio.transforms import MFCC


def load_audio(file_path, target_sr=16000):
    signal, sr = librosa.load(file_path, sr=target_sr)
    return signal, sr


def normalize_audio(signal):
    return librosa.util.normalize(signal)


def extract_mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.T  # Transpose to get time as the first dimension


def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def extract_and_normalize_features(signal, sr):
    # Extract MFCCs
    mfccs = extract_mfcc(signal, sr)

    # Extract spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]

    # Extract spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]

    # Combine features
    features = np.column_stack((mfccs, spectral_centroids, spectral_bandwidth))

    # Normalize features
    normalized_features = normalize_features(features)

    return normalized_features


def resample_features(features, target_length):
    original_length = features.shape[0]
    feature_dim = features.shape[1]

    # Create an interpolation function for each feature dimension
    interpolators = [interp1d(np.arange(original_length), features[:, i]) for i in range(feature_dim)]

    # Create new time points
    new_time_points = np.linspace(0, original_length - 1, target_length)

    # Interpolate each feature dimension
    resampled_features = np.column_stack([interp(new_time_points) for interp in interpolators])

    return resampled_features


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
        speaker = f"[SPEAKER_{segment['speaker']}]"
        text = segment['text']
        vtt_output += f"{start} --> {end}\n{speaker}: {text}\n\n"
    return vtt_output


def extract_embeddings_wav2vec2(signal, sr, segment_length, overlap):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    model_name = "Edresson/wav2vec2-large-100k-voxpopuli-ft-Common-Voice_plus_TTS-Dataset-russian"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    step = segment_length - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than segment length")
    num_segments = int(np.ceil(len(signal) / sr / step))

    def process_segment(i):
        start = int(i * step * sr)
        end = start + int(segment_length * sr)
        segment = signal[start:end]

        if len(segment) == 0:
            return None

        inputs = processor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the mean of the last hidden state as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding, (start / sr, end / sr)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_segment, range(num_segments)))

    embeddings, times = zip(*[r for r in results if r is not None])
    return np.array(embeddings), times


def extract_embeddings_speechbrain(signal, sr, segment_length, overlap):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")

    step = segment_length - overlap
    if step <= 0:
        raise ValueError("Overlap must be less than segment length")

    num_segments = int(np.ceil((len(signal) / sr - overlap) / step))

    def process_segment(i):
        start = i * step
        end = start + segment_length
        segment = signal[int(start * sr):int(end * sr)]
        if len(segment) > 0:
            embedding = classifier.encode_batch(torch.tensor(segment).unsqueeze(0)).squeeze().numpy()
            return embedding, (start, min(end, len(signal) / sr))
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_segment, range(num_segments)))

    embeddings, times = zip(*[r for r in results if r is not None])
    return np.array(embeddings), times


def visualize_clusters(embeddings, labels, title):
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')

    # Add a color bar
    plt.colorbar(scatter)

    # Set title and labels
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    # Show the plot
    plt.show()


def time_constrained_spectral_clustering(embeddings, timestamps, n_clusters=10, max_time_diff=5.0):
    n_samples = len(embeddings)
    affinity_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            time_diff = timestamps[j][0] - timestamps[i][1]  # Start of j - End of i
            if time_diff <= max_time_diff:
                similarity = 1 / (1 + np.linalg.norm(embeddings[i] - embeddings[j]))
                affinity_matrix[i, j] = affinity_matrix[j, i] = similarity

    clustering = SpectralClustering(n_clusters=n_clusters,
                                    affinity='precomputed',
                                    random_state=42)
    labels = clustering.fit_predict(affinity_matrix)
    return labels


def refine_clusters(spectral_labels, embeddings, min_cluster_size=5, distance_threshold=1.5):
    clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
    refined_labels = clustering.fit_predict(embeddings)

    combined_labels = spectral_labels.copy()
    unique_labels, counts = np.unique(refined_labels, return_counts=True)
    large_clusters = unique_labels[counts >= min_cluster_size]
    small_clusters = unique_labels[counts < min_cluster_size]

    large_cluster_points = embeddings[np.isin(refined_labels, large_clusters)]
    small_cluster_indices = np.where(np.isin(refined_labels, small_clusters))[0]

    if len(large_cluster_points) == 0:
        return spectral_labels  # Если нет точек в больших кластерах, вернуть исходные метки

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(large_cluster_points)

    for idx in small_cluster_indices:
        point = embeddings[idx].reshape(1, -1)
        _, nearest_idx = nn.kneighbors(point)
        nearest_label = refined_labels[np.isin(refined_labels, large_clusters)][nearest_idx[0][0]]
        combined_labels[idx] = nearest_label

    return combined_labels


def perform_clustering(embeddings, timestamps, n_clusters=10, max_time_diff=5.0, min_cluster_size=5, distance_threshold=1.5):
    # Time-constrained Spectral Clustering
    spectral_labels = time_constrained_spectral_clustering(embeddings, timestamps, n_clusters, max_time_diff)

    # Refine clusters based on Agglomerative Clustering
    refined_labels = refine_clusters(spectral_labels, embeddings, min_cluster_size, distance_threshold)

    # Further reduce the number of clusters by merging small clusters
    unique_labels, counts = np.unique(refined_labels, return_counts=True)
    small_clusters = unique_labels[counts < min_cluster_size]
    large_clusters = unique_labels[counts >= min_cluster_size]

    if len(large_clusters) == 0:
        return refined_labels  # Если нет больших кластеров, вернуть исходные метки

    large_cluster_points = embeddings[np.isin(refined_labels, large_clusters)]
    small_cluster_indices = np.where(np.isin(refined_labels, small_clusters))[0]

    if len(large_cluster_points) == 0:
        return refined_labels  # Если нет точек в больших кластерах, вернуть исходные метки

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(large_cluster_points)

    for idx in small_cluster_indices:
        point = embeddings[idx].reshape(1, -1)
        _, nearest_idx = nn.kneighbors(point)
        nearest_label = refined_labels[np.isin(refined_labels, large_clusters)][nearest_idx[0][0]]
        refined_labels[idx] = nearest_label

    return refined_labels


def find_optimal_clusters_elbow(embeddings, max_clusters=15):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=600, n_init=10, random_state=0)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)

    kl = KneeLocator(range(1, max_clusters + 1), wcss, curve='convex', direction='decreasing')
    return kl.elbow


def reassign_small_clusters(embeddings, labels, min_cluster_size=5):
    unique_labels, counts = np.unique(labels, return_counts=True)
    large_clusters = unique_labels[counts >= min_cluster_size]
    small_clusters = unique_labels[counts < min_cluster_size]

    if len(large_clusters) == 0:
        return labels  # Если нет больших кластеров, вернуть исходные метки

    large_cluster_points = embeddings[np.isin(labels, large_clusters)]
    small_cluster_indices = np.where(np.isin(labels, small_clusters))[0]

    if len(large_cluster_points) == 0:
        return labels  # Если нет точек в больших кластерах, вернуть исходные метки

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(large_cluster_points)

    for idx in small_cluster_indices:
        point = embeddings[idx].reshape(1, -1)
        _, nearest_idx = nn.kneighbors(point)
        nearest_label = labels[np.isin(labels, large_clusters)][nearest_idx[0][0]]
        labels[idx] = nearest_label

    return labels


def smooth_labels(labels, times, window_size=5):
    smoothed_labels = labels.copy()
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2 + 1)
        window_labels = labels[start:end]
        smoothed_labels[i] = np.argmax(np.bincount(window_labels))
    return smoothed_labels


def merge_similar_clusters(embeddings, labels, threshold=0.5):
    unique_labels = np.unique(labels)
    centroids = np.array([embeddings[labels == l].mean(axis=0) for l in unique_labels])
    distances = squareform(pdist(centroids))

    new_labels = labels.copy()
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            if distances[i, j] < threshold:
                new_labels[new_labels == unique_labels[j]] = unique_labels[i]

    return new_labels

# def perform_clustering(embeddings, distance_threshold=1.5, max_clusters=10):
#     # Agglomerative Clustering
#     agg_clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
#     agg_labels = agg_clustering.fit_predict(embeddings)
#
#     # Function to merge similar clusters
#     def merge_similar_clusters(embeddings, labels, threshold):
#         unique_labels = np.unique(labels)
#         centroids = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
#
#         while True:
#             distances = np.linalg.norm(centroids[:, None] - centroids, axis=2)
#             np.fill_diagonal(distances, np.inf)
#             closest_pair = np.unravel_index(distances.argmin(), distances.shape)
#
#             if distances[closest_pair] > threshold:
#                 break
#
#             # Merge clusters
#             old_label, new_label = unique_labels[list(closest_pair)]
#             labels[labels == old_label] = new_label
#
#             # Update centroids
#             unique_labels = np.unique(labels)
#             centroids = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
#
#         return labels
#
#     # Merge similar clusters
#     merged_labels = merge_similar_clusters(embeddings, agg_labels, distance_threshold)
#
#     # Gaussian Mixture Model for comparison
#     n_components_range = range(1, min(max_clusters, len(np.unique(merged_labels)) + 1))
#     gmm_models = [GaussianMixture(n_components=n, random_state=42).fit(embeddings) for n in n_components_range]
#     gmm_bic = [model.bic(embeddings) for model in gmm_models]
#     best_gmm = gmm_models[np.argmin(gmm_bic)]
#     gmm_labels = best_gmm.predict(embeddings)
#
#     # Choose the clustering with better silhouette score
#     sil_score_merged = silhouette_score(embeddings, merged_labels) if len(np.unique(merged_labels)) > 1 else -1
#     sil_score_gmm = silhouette_score(embeddings, gmm_labels) if len(np.unique(gmm_labels)) > 1 else -1
#
#     final_labels = merged_labels if sil_score_merged >= sil_score_gmm else gmm_labels
#
#     visualize_clusters(embeddings, final_labels, "Final Clustering Results")
# visualize_clusters(embeddings, merged_labels, "Merged Agglomerative Clustering Results")
# visualize_clusters(embeddings, gmm_labels, "Gaussian Mixture Model Results")

# return final_labels
# # Agglomerative Clustering
# clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None)
# agglomerative_labels = clustering.fit_predict(embeddings)
#
# visualize_clusters(embeddings, agglomerative_labels, "Agglomerative Results")
#
# return agglomerative_labels


def main(wav_file_path, vtt_file_path, output_vtt_file_path):
    # Load audio file
    signal, sr = load_audio(wav_file_path)

    # Apply audio normalization
    normalized_signal = normalize_audio(signal)

    # Extract and normalize features
    features = extract_and_normalize_features(normalized_signal, sr)

    # Extract embeddings using wav2vec2
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_wav2vec2 = executor.submit(extract_embeddings_wav2vec2, signal, sr, segment_length=2, overlap=1.2)
        embeddings_wav2vec2, times_wav2vec2 = future_wav2vec2.result()
        future_speechbrain = executor.submit(extract_embeddings_speechbrain, signal, sr, segment_length=2.0,
                                             overlap=1.0)
        embeddings_speechbrain, times_speechbrain = future_speechbrain.result()
    print(
        f"Extracted {len(embeddings_wav2vec2)} embeddings from wav2vec2 and {len(embeddings_speechbrain)} from "
        f"speechbrain.")

    # Resample features to match the number of embeddings
    target_length = min(len(embeddings_wav2vec2), len(embeddings_speechbrain))
    resampled_features = resample_features(features, target_length)

    min_length = min(len(embeddings_wav2vec2), len(embeddings_speechbrain))
    embeddings_wav2vec2 = embeddings_wav2vec2[:min_length]
    embeddings_speechbrain = embeddings_speechbrain[:min_length]
    times_wav2vec2 = times_wav2vec2[:min_length]

    if embeddings_wav2vec2.shape[1] != embeddings_speechbrain.shape[1]:
        embeddings_speechbrain = np.resize(embeddings_speechbrain, (min_length, embeddings_wav2vec2.shape[1]))

    combined_embeddings = np.concatenate((embeddings_wav2vec2, embeddings_speechbrain, resampled_features), axis=1)
    print(f"Combined {len(combined_embeddings)} embeddings.")

    # Determine optimal number of clusters
    optimal_clusters_elbow = find_optimal_clusters_elbow(combined_embeddings)

    print(f"Optimal clusters (Elbow method): {optimal_clusters_elbow}")

    # Choose one of the methods or take an average
    n_clusters = optimal_clusters_elbow

    # Perform clustering
    labels = perform_clustering(combined_embeddings,
                                times_wav2vec2,
                                n_clusters=n_clusters,
                                max_time_diff=7.0,
                                distance_threshold=3.0)

    # Apply post-processing
    labels = reassign_small_clusters(combined_embeddings, labels)
    labels = smooth_labels(labels, times_wav2vec2)
    labels = merge_similar_clusters(combined_embeddings, labels, threshold=1.9)

    print(f"Generated {len(set(labels))} unique speaker labels.")

    # Visualize results (you may need to adjust your visualization code)
    visualize_clusters(combined_embeddings, labels, "Similar clusters")

    # Load VTT file and match speakers to segments
    captions = load_vtt(vtt_file_path)
    matched_segments = match_speakers_to_segments(captions, labels, times_wav2vec2)

    # Postprocess segments
    postprocessed_segments = postprocess_segments(matched_segments)

    # Format and save VTT file
    vtt_output = format_vtt(postprocessed_segments)
    with open(output_vtt_file_path, 'w', encoding='utf-8') as f:
        f.write(vtt_output)
    print(f"VTT file saved to {output_vtt_file_path}")


# Example usage
# wav_file_path = r"D:\Projects\VideoTransl\dev\dev.mp4.vocals.wav"
# vtt_file_path = r"D:\Projects\VideoTransl\dev\16hzconv.vtt"
# output_vtt_file_path = r"D:\Projects\VideoTransl\dev\output.vtt"

wav_file_path = r"D:\Projects\VideoTransl\dev\dev10m.mp4.vocals.wav"
vtt_file_path = r"D:\Projects\VideoTransl\dev\16hzconv.vtt"
output_vtt_file_path = r"D:\Projects\VideoTransl\dev\output.vtt"

# wav_file_path = r"D:\Projects\VideoTransl\rus\source.mp4.vocals.wav"
# vtt_file_path = r"D:\Projects\VideoTransl\rus\1.mp4.vocals.vtt"
# output_vtt_file_path = r"D:\Projects\VideoTransl\rus\output.vtt"
main(wav_file_path, vtt_file_path, output_vtt_file_path)
