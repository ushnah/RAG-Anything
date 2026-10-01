"""Reference-gallery face identification for sampled media frames."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


class FaceIdentifier:
    """Identify faces against a local reference gallery.

    InsightFace is loaded lazily. A missing model, empty gallery, or failed frame
    identification disables this enrichment without interrupting ingestion.
    """

    def __init__(self, gallery_path, model_name='buffalo_l', threshold=0.5,
                 min_face_size=40, min_detection_score=0.6, app=None):
        self.gallery_path = Path(gallery_path) if gallery_path else None
        self.model_name = model_name
        self.threshold = float(threshold)
        self.min_face_size = int(min_face_size)
        self.min_detection_score = float(min_detection_score)
        self.app = app
        self.gallery = []
        self.enabled = bool(self.gallery_path)
        if self.enabled:
            self._load_backend()
            if self.app is not None:
                self.build_gallery()

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Face embedding is empty or invalid')
        return vector / norm

    def _load_backend(self):
        if self.app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as exc:
            self.enabled = False
            self.app = None
            logger.warning('Face identification disabled: could not load InsightFace: %s', exc)

    def _gallery_files(self):
        if not self.gallery_path or not self.gallery_path.is_dir():
            return []
        return sorted(path for path in self.gallery_path.rglob('*')
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

    def _gallery_signature(self, files):
        digest = hashlib.sha256()
        for path in files:
            stat = path.stat()
            digest.update(str(path.relative_to(self.gallery_path)).encode())
            digest.update(f'{stat.st_size}:{stat.st_mtime_ns}'.encode())
        return digest.hexdigest()

    def _cache_path(self):
        return self.gallery_path / '.face_gallery.npz'

    def build_gallery(self):
        """Load cached embeddings or build one embedding per valid reference image."""
        if not self.enabled or self.app is None:
            return self.gallery
        files = self._gallery_files()
        if not files:
            self.gallery = []
            logger.warning('Face gallery is empty: %s', self.gallery_path)
            return self.gallery
        signature = self._gallery_signature(files)
        cache = self._cache_path()
        if cache.is_file():
            try:
                data = np.load(cache, allow_pickle=False)
                if str(data['signature']) == signature and str(data['model_name']) == self.model_name:
                    names = data['names'].tolist()
                    embeddings = data['embeddings']
                    self.gallery = list(zip(names, embeddings))
                    logger.info('Loaded %d cached face references from %s', len(self.gallery), cache)
                    return self.gallery
            except (OSError, ValueError, KeyError):
                logger.warning('Ignoring invalid face gallery cache: %s', cache)

        entries = []
        counts = {}
        for path in files:
            person = path.parent.relative_to(self.gallery_path).parts[0]
            try:
                import cv2
                image = cv2.imread(str(path))
                faces = self.app.get(image) if image is not None else []
                if len(faces) == 0:
                    logger.warning('Skipped %s: no face detected', path)
                    continue
                if len(faces) != 1:
                    logger.warning('Skipped %s: multiple faces detected', path)
                    continue
                face = faces[0]
                self._validate_face(face)
                entries.append((person, self.normalize_embedding(face.embedding)))
                counts[person] = counts.get(person, 0) + 1
            except Exception as exc:
                logger.warning('Skipped %s: %s', path, exc)
        self.gallery = entries
        if entries:
            np.savez_compressed(self._cache_path(),
                                signature=signature, model_name=self.model_name,
                                names=np.array([name for name, _ in entries]),
                                embeddings=np.array([embedding for _, embedding in entries], dtype=np.float32))
        for person, count in sorted(counts.items()):
            logger.info('Loaded %d reference images for %s', count, person)
        return self.gallery

    def _validate_face(self, face):
        score = getattr(face, 'det_score', 1.0)
        bbox = np.asarray(getattr(face, 'bbox', [0, 0, 0, 0]))
        if score < self.min_detection_score:
            raise ValueError(f'detection score {score:.3f} below minimum')
        if bbox.size < 4 or min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < self.min_face_size:
            raise ValueError('face is below the minimum size')
        if not hasattr(face, 'embedding'):
            raise ValueError('face has no embedding')

    def identify_face(self, face_embedding):
        """Return the best match, or an explicit unknown result."""
        unknown = {'name': None, 'similarity': None, 'source': 'face_recognition'}
        if not self.gallery:
            return unknown
        embedding = self.normalize_embedding(face_embedding)
        best_name, best_score = None, -1.0
        for name, reference in self.gallery:
            score = float(np.dot(embedding, reference))
            if score > best_score:
                best_name, best_score = name, score
        if best_score < self.threshold:
            unknown['similarity'] = round(best_score, 4)
            return unknown
        return {'name': best_name, 'similarity': round(best_score, 4),
                'source': 'face_recognition'}

    def identify_faces(self, frame):
        """Identify all usable faces in an image path or OpenCV image."""
        if not self.enabled or self.app is None or not self.gallery:
            return []
        try:
            import cv2
            image = cv2.imread(str(frame)) if isinstance(frame, (str, Path)) else frame
            if image is None:
                logger.warning('Face identification skipped: could not read %s', frame)
                return []
            results = []
            for face in self.app.get(image):
                try:
                    self._validate_face(face)
                    results.append(self.identify_face(face.embedding))
                except (TypeError, ValueError) as exc:
                    logger.info('Skipped low-quality detected face: %s', exc)
            return results
        except Exception as exc:
            logger.warning('Face identification failed for %s: %s', frame, exc)
            return []
