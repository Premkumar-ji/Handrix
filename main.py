import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=2
        )
        
        # Screen setup
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        
        # Control parameters
        self.MOUSE_SMOOTHING = 3
        self.CLICK_DISTANCE = 0.025
        self.FIST_THRESHOLD = 0.15
        self.SCROLL_SPEED = 80
        self.DEBOUNCE_TIME = 0.3
        self.HAND_CONTACT_THRESHOLD = 0.1
        
        # State tracking
        self.prev_x, self.prev_y = pyautogui.position()
        self.last_right_click = 0
        self.last_left_click = 0
        self.last_scroll_time = 0
        self.system_enabled = True
        self.last_toggle_time = 0
        self.last_scroll_toggle_time = 0
        self.scroll_active = False
        
    def get_hand_openness(self, hand_landmarks, frame_shape):
        """Calculate how closed the hand is (0=fist, 1=open)"""
        h, w = frame_shape[:2]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        fingertips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        avg_distance = np.mean([np.sqrt((tip.x*w - wrist.x*w)**2 + (tip.y*h - wrist.y*h)**2) for tip in fingertips])
        return np.clip(avg_distance / 200, 0, 1)
    
    def is_thumb_up(self, hand_landmarks):
        """Check if thumb is extended up"""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        return thumb_tip.y < thumb_mcp.y
    
    def is_scroll_toggle_gesture(self, hand_landmarks):
        """Check if hand is making the scroll toggle gesture"""
        # Get all finger tip positions
        fingertips = {
            'thumb': hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            'index': hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            'middle': hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            'ring': hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            'pinky': hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        }
        
        # Get corresponding joint positions (one before tip)
        finger_joints = {
            'thumb': hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP],
            'index': hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP],
            'middle': hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
            'ring': hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP],
            'pinky': hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP]
        }
        
        # Check each finger position
        is_finger_up = {}
        for finger in fingertips:
            is_finger_up[finger] = fingertips[finger].y < finger_joints[finger].y
        
        # Check for scroll toggle gesture
        # Index, Middle, and Thumb up, Ring and Pinky down
        return (is_finger_up['index'] and 
                is_finger_up['middle'] and 
                is_finger_up['thumb'] and 
                not is_finger_up['ring'] and 
                not is_finger_up['pinky'])
    
    def process_frame(self):
        """Process a single frame and return the processed frame"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        
        # Process hand landmarks
        results = self.hands.process(rgb_frame)
        right_hand_pos = None
        left_hand_pos = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if not results.multi_handedness:
                    continue
                    
                hand_label = results.multi_handedness[idx].classification[0].label
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                distance = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                
                # Store hand positions for system toggle
                if hand_label == "Right":
                    right_hand_pos = (x, y)
                elif hand_label == "Left":
                    left_hand_pos = (x, y)
                
                # Right Hand - Mouse Control
                if hand_label == "Right" and self.system_enabled:
                    if distance < self.CLICK_DISTANCE and (current_time - self.last_right_click) > self.DEBOUNCE_TIME:
                        pyautogui.rightClick()
                        self.last_right_click = current_time
                        cv2.putText(frame, "RIGHT CLICK", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if (current_time - self.last_right_click) > 0.1:
                        x_scaled = np.interp(x, [int(w*0.05), int(w*0.95)], [0, self.screen_width])
                        y_scaled = np.interp(y, [int(h*0.05), int(h*0.95)], [0, self.screen_height])
                        smoothed_x = self.prev_x + (x_scaled - self.prev_x) / self.MOUSE_SMOOTHING
                        smoothed_y = self.prev_y + (y_scaled - self.prev_y) / self.MOUSE_SMOOTHING
                        pyautogui.moveTo(smoothed_x, smoothed_y)
                        self.prev_x, self.prev_y = smoothed_x, smoothed_y
                
                # Left Hand - Click and Scroll Control
                elif hand_label == "Left" and self.system_enabled:
                    # Left click
                    if distance < self.CLICK_DISTANCE and (current_time - self.last_left_click) > self.DEBOUNCE_TIME:
                        pyautogui.click()
                        self.last_left_click = current_time
                        cv2.putText(frame, "LEFT CLICK", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Scroll control
                    if self.is_scroll_toggle_gesture(hand_landmarks):
                        time_since_last_scroll_toggle = current_time - self.last_scroll_toggle_time
                        
                        # Only toggle scroll if 2 seconds have passed since last scroll toggle
                        if time_since_last_scroll_toggle >= 2.0:
                            self.scroll_active = not self.scroll_active
                            self.last_scroll_toggle_time = current_time
                            
                            if self.scroll_active:
                                cv2.putText(frame, "SCROLL MODE ACTIVE", (w//2-100, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            else:
                                cv2.putText(frame, "SCROLL MODE DISABLED", (w//2-100, 150),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        else:
                            # Show remaining time to wait
                            remaining_time = 2.0 - time_since_last_scroll_toggle
                            cv2.putText(frame, f"Wait {remaining_time:.1f}s", (w//2-100, 120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # If scroll is active, handle scrolling
                    if self.scroll_active:
                        if (current_time - self.last_scroll_time) > 0.1:
                            if self.is_thumb_up(hand_landmarks):
                                pyautogui.scroll(self.SCROLL_SPEED)
                                cv2.putText(frame, "SCROLL UP", (x, y-80),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                pyautogui.scroll(-self.SCROLL_SPEED)
                                cv2.putText(frame, "SCROLL DOWN", (x, y-80),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            self.last_scroll_time = current_time
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, hand_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # System toggle
        if (right_hand_pos and left_hand_pos and 
            (current_time - self.last_toggle_time) > 1.0):
            contact_dist = np.hypot(right_hand_pos[0]-left_hand_pos[0],
                                  right_hand_pos[1]-left_hand_pos[1]) / max(w, h)
            if contact_dist < self.HAND_CONTACT_THRESHOLD:
                self.system_enabled = not self.system_enabled
                self.last_toggle_time = current_time
                cv2.circle(frame, right_hand_pos, 50, (0, 255, 255), 5)
                cv2.circle(frame, left_hand_pos, 50, (0, 255, 255), 5)
        
        # Display status
        status_color = (0, 255, 0) if self.system_enabled else (0, 0, 255)
        status_text = "SYSTEM: ON" if self.system_enabled else "SYSTEM: OFF"
        cv2.putText(frame, status_text, (w//2-100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
        
        # Display scroll mode
        if self.scroll_active:
            cv2.putText(frame, "SCROLL ACTIVE", (w//2-100, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop to run the hand gesture control system"""
        try:
            while self.cap.isOpened():
                frame = self.process_frame()
                if frame is None:
                    break
                
                cv2.imshow("Precision Hand Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()