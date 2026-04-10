import cv2
import mediapipe as mp
import time
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image
import numpy as np
import os
import urllib.request


MODEL_FILE = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading {MODEL_FILE}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Model downloaded!")


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        """
        Initialise le hand tracker avec MediaPipe.
        
        Args:
            max_hands: Nombre maximum de mains à détecter
            detection_confidence: Seuil de confiance pour la détection
            tracking_confidence: Seuil de confiance pour le tracking
        """
        ensure_model()
        
        base_options = BaseOptions(model_asset_path=MODEL_FILE)
        options = HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.hands = HandLandmarker.create_from_options(options)

        # Couleurs personnalisées pour chaque doigt (BGR)
        self.finger_colors = {
            'pouce':       (0, 255, 255),   # Jaune
            'index':       (0, 255, 0),     # Vert
            'majeur':      (255, 0, 0),     # Bleu
            'annulaire':   (0, 0, 255),     # Rouge
            'auriculaire': (255, 0, 255),   # Magenta
            'paume':       (255, 255, 255)  # Blanc
        }

        # Définition des bones (connexions) de la main
        # Chaque bone est défini par (landmark_debut, landmark_fin, nom_du_doigt)
        self.bones = [
            # Pouce
            (0, 1, 'pouce'),
            (1, 2, 'pouce'),
            (2, 3, 'pouce'),
            (3, 4, 'pouce'),

            # Index
            (0, 5, 'index'),
            (5, 6, 'index'),
            (6, 7, 'index'),
            (7, 8, 'index'),

            # Majeur
            (0, 9, 'majeur'),
            (9, 10, 'majeur'),
            (10, 11, 'majeur'),
            (11, 12, 'majeur'),

            # Annulaire
            (0, 13, 'annulaire'),
            (13, 14, 'annulaire'),
            (14, 15, 'annulaire'),
            (15, 16, 'annulaire'),

            # Auriculaire
            (0, 17, 'auriculaire'),
            (17, 18, 'auriculaire'),
            (18, 19, 'auriculaire'),
            (19, 20, 'auriculaire'),

            # Connexions de la paume (entre les bases des doigts)
            (5, 9, 'paume'),
            (9, 13, 'paume'),
            (13, 17, 'paume'),
        ]

        # Noms des landmarks
        self.landmark_names = {
            0: "WRIST",
            1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
            5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
            9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
            13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
            17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
        }

        self.prev_time = 0
        self.state = "IDLE"
        self.start_point = None
        self.traces = {}

    def detect_gesture(self, results):
        """
        Détecte les gestes de la main.
        
        Returns:
            state: 'PINCH', 'GRAB', ou 'IDLE'
        """
        if not results.hand_landmarks:
            self.state = "IDLE"
            return self.state
        
        hand_landmarks = results.hand_landmarks[0]
        h, w = 1.0, 1.0
        
        landmarks = {
            i: (hand_landmarks[i].x, hand_landmarks[i].y, hand_landmarks[i].z) 
            for i in range(21)
        }
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        pinch_dist = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2) ** 0.5
        pinch_threshold = 0.05
        
        if pinch_dist < pinch_threshold:
            self.state = "PINCH"
            return self.state
        
        fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        distances = []
        for i in range(len(fingertips) - 1):
            d = ((fingertips[i][0] - fingertips[i+1][0])**2 + (fingertips[i][1] - fingertips[i+1][1])**2) ** 0.5
            distances.append(d)
        
        avg_tip_dist = sum(distances) / len(distances)
        tip_dist_threshold = 0.08
        
        if avg_tip_dist < tip_dist_threshold:
            self.state = "GRAB"
            return self.state
        
        self.state = "IDLE"
        return self.state

    def update_traces(self, frame, results):
        """
        Met à jour les tracés en fonction de l'état.
        
        PINCH = trace bleu, GRAB = trace rouge
        """
        h, w, _ = frame.shape
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            wrist = hand_landmarks[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            
            if self.state == "PINCH":
                if self.start_point is None:
                    self.start_point = (cx, cy)
                    self.traces["pinch"] = [(cx, cy)]
                else:
                    dx = cx - self.start_point[0]
                    dy = cy - self.start_point[1]
                    if (dx**2 + dy**2) ** 0.5 > 5:
                        if "pinch" not in self.traces:
                            self.traces["pinch"] = []
                        self.traces["pinch"].append((cx, cy))
                        self.start_point = (cx, cy)
                        if len(self.traces["pinch"]) > 50:
                            self.traces["pinch"].pop(0)
            
            elif self.state == "GRAB":
                if self.start_point is None:
                    self.start_point = (cx, cy)
                    self.traces["grab"] = [(cx, cy)]
                else:
                    dx = cx - self.start_point[0]
                    dy = cy - self.start_point[1]
                    if (dx**2 + dy**2) ** 0.5 > 5:
                        if "grab" not in self.traces:
                            self.traces["grab"] = []
                        self.traces["grab"].append((cx, cy))
                        self.start_point = (cx, cy)
                        if len(self.traces["grab"]) > 50:
                            self.traces["grab"].pop(0)
            
            else:
                self.start_point = None
        else:
            self.start_point = None
        
        self.draw_traces(frame)
        return frame

    def draw_traces(self, frame):
        """
        Dessine les tracés sur la frame.
        """
        if "pinch" in self.traces and len(self.traces["pinch"]) > 1:
            for i in range(1, len(self.traces["pinch"])):
                pt1 = self.traces["pinch"][i-1]
                pt2 = self.traces["pinch"][i]
                cv2.line(frame, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
        
        if "grab" in self.traces and len(self.traces["grab"]) > 1:
            for i in range(1, len(self.traces["grab"])):
                pt1 = self.traces["grab"][i-1]
                pt2 = self.traces["grab"][i]
                cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        
        return frame

    def find_hands(self, frame):
        """
        Détecte les mains dans une frame.
        
        Args:
            frame: Image BGR d'OpenCV
            
        Returns:
            results: Résultats de la détection MediaPipe
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.hands.detect(mp_image)
        return results

    def draw_bones(self, frame, results, show_labels=True):
        """
        Dessine les bones (squelette) de la main sur la frame.
        
        Args:
            frame: Image BGR d'OpenCV
            results: Résultats de la détection MediaPipe
            show_labels: Afficher les noms des landmarks
            
        Returns:
            frame: Image avec les bones dessinés
        """
        h, w, _ = frame.shape

        if results.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):

                handedness = "Droite"
                confidence = 0.0
                if results.handedness and hand_idx < len(results.handedness):
                    handedness_info = results.handedness[hand_idx]
                    label = handedness_info[0].category_name
                    handedness = "Gauche" if label == "Left" else "Droite"
                    confidence = handedness_info[0].score

                landmarks_px = {}
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cz = lm.z
                    landmarks_px[idx] = (cx, cy, cz)

                for start_idx, end_idx, finger_name in self.bones:
                    if start_idx in landmarks_px and end_idx in landmarks_px:
                        pt1 = landmarks_px[start_idx][:2]
                        pt2 = landmarks_px[end_idx][:2]
                        color = self.finger_colors.get(finger_name, (255, 255, 255))

                        avg_z = (landmarks_px[start_idx][2] + landmarks_px[end_idx][2]) / 2
                        thickness = max(1, int(3 - avg_z * 10))
                        thickness = min(thickness, 6)

                        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

                for idx, (cx, cy, cz) in landmarks_px.items():
                    radius = max(3, int(5 - cz * 10))
                    radius = min(radius, 10)

                    tips = [4, 8, 12, 16, 20]
                    if idx in tips:
                        cv2.circle(frame, (cx, cy), radius + 4, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.circle(frame, (cx, cy), radius + 2, (0, 200, 255), -1, cv2.LINE_AA)
                    elif idx == 0:
                        cv2.circle(frame, (cx, cy), radius + 3, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (cx, cy), radius + 3, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (cx, cy), radius, (50, 50, 50), -1, cv2.LINE_AA)
                        cv2.circle(frame, (cx, cy), radius, (200, 200, 200), 1, cv2.LINE_AA)

                    if show_labels and idx in [0, 4, 8, 12, 16, 20]:
                        label_text = self.landmark_names.get(idx, str(idx))
                        cv2.putText(
                            frame, label_text,
                            (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (255, 255, 255), 1, cv2.LINE_AA
                        )

                wrist = landmarks_px.get(0, None)
                if wrist:
                    cv2.putText(
                        frame, f"Main {handedness} ({confidence:.0%})",
                        (wrist[0] - 50, wrist[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA
                    )

        return frame

    def draw_info_panel(self, frame, results):
        """
        Dessine un panneau d'informations en haut à gauche.
        """
        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Nombre de mains
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        cv2.putText(frame, f"Mains detectees: {num_hands}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # State (PINCH/GRAB/IDLE)
        cv2.putText(frame, f"State: {self.state}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Légende des couleurs
        y_offset = 85
        for finger, color in self.finger_colors.items():
            cv2.circle(frame, (30, y_offset), 6, color, -1)
            cv2.putText(frame, finger.capitalize(), (45, y_offset + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18

        return frame

    def draw_builtin_style(self, frame, results):
        """
        Dessine avec le style intégré de MediaPipe (alternative).
        """
        return frame

    def release(self):
        """Libère les ressources."""
        self.hands.close()


def main():
    """Fonction principale."""
    print("=" * 50)
    print("  HAND TRACKING - OpenCV + MediaPipe")
    print("=" * 50)
    print()
    print("Controles:")
    print("  [Q] / [ESC] - Quitter")
    print("  [L] - Activer/Desactiver les labels")
    print("  [S] - Changer de style (custom/mediapipe)")
    print("  [M] - Activer/Desactiver le miroir")
    print()

    # Initialiser la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la camera!")
        return

    # Configurer la résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialiser le tracker
    tracker = HandTracker(max_hands=2, detection_confidence=0.7, tracking_confidence=0.7)

    # Options d'affichage
    show_labels = True
    use_custom_style = True
    mirror = True

    print("Camera ouverte. En attente de mains...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire la frame!")
            break

        # Miroir (flip horizontal)
        if mirror:
            frame = cv2.flip(frame, 1)

        # Détection des mains
        results = tracker.find_hands(frame)
        
        # Détection du geste
        state = tracker.detect_gesture(results)
        
        # Mise à jour des tracés
        tracker.update_traces(frame, results)

        # Dessiner les bones
        if use_custom_style:
            frame = tracker.draw_bones(frame, results, show_labels=show_labels)
        else:
            frame = tracker.draw_builtin_style(frame, results)

        # Panneau d'infos
        frame = tracker.draw_info_panel(frame, results)

        # Instructions en bas
        cv2.putText(
            frame,
            "[Q]uit  [L]abels  [S]tyle  [M]iroir",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (150, 150, 150), 1, cv2.LINE_AA
        )

        # Afficher
        cv2.imshow("Hand Tracking", frame)

        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q ou ESC
            break
        elif key == ord('l'):
            show_labels = not show_labels
            print(f"Labels: {'ON' if show_labels else 'OFF'}")
        elif key == ord('s'):
            use_custom_style = not use_custom_style
            print(f"Style: {'Custom' if use_custom_style else 'MediaPipe'}")
        elif key == ord('m'):
            mirror = not mirror
            print(f"Miroir: {'ON' if mirror else 'OFF'}")

    # Nettoyage
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Programme terminé.")


if __name__ == "__main__":
    main()