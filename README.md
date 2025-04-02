# Handrix
ChatGPT said: HandriX is an AI-powered hand gesture recognition system enabling touchless control for mouse, clicks, scrolling, and system toggling using intuitive movements.


HandriX - AI-Powered Hand Gesture Control System
Gesture Mapping:
Right Hand:

Moves freely to control the mouse cursor.

Right-click is performed by a specific right-hand gesture.

Left Hand:

Index + Thumb Touch: Triggers a left-click.

Scroll Mode Activation:

Form gesture: Index, Middle, and Thumb up; Ring and Pinky down.

Once in scroll mode, the Thumb Up/Down controls vertical scrolling.

Both Hands Together:

Activates or toggles the system on/off.

Keyboard Escape ('q'):

Pressing 'q' exits the program.

Implementation Flow
Gesture Recognition Model:

Uses computer vision (OpenCV & MediaPipe) for hand tracking.

Implements a Machine Learning model to recognize specific gestures.

Mouse & Scroll Control Integration:

Python’s pyautogui library controls the cursor, clicks, and scrolling.

Dynamic gesture-state switching ensures smooth input handling.

System Activation & Deactivation:

Detects when both hands come together to toggle the system.

Performance Optimization:

Uses real-time gesture detection and minimal lag response.

Can run efficiently on mid-range hardware.

Potential Enhancements:
Gesture Customization: Allow users to define their own gestures.

Voice Integration: Combine voice commands with gestures.

3D Hand Tracking: Enhance precision with depth-sensing cameras.
