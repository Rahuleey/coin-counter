from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO('final_model_last.pt')

# Define a dictionary to map class labels to their corresponding coin values
coin_values = {
    "1_rupeescoin": 1,
    "2_rupeescoin": 2,
    "5_rupeescoin": 5,
    "10_rupeescoin": 10,
    "20_rupeescoin": 20
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the font and background color
font = cv2.FONT_HERSHEY_TRIPLEX
bg_color = (160,32,240)  # Black color for the background

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Run inference on the frame
        results = model(frame, stream=True)

        # Initialize variables to store the total number of coins and the total sum
        total_coins = 0
        total_sum = 0

        # Loop through the detected objects
        for result in results:
            detection = result.boxes

            for box in detection:
                class_id = box.cls[0].tolist()  # Get the scalar value from the tensor
                class_name = model.names[int(class_id)]

                # Check if the class name is in the coin_values dictionary
                if class_name in coin_values:
                    total_coins += 1
                    total_sum += coin_values[class_name]
                    # Draw the bounding box and label on the frame
                    x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (36, 255, 12), 2)

        # Get the size of the text for the total number of coins and the total sum
        coins_text = f"No_of_coins: {total_coins}"
        sum_text = f"Sum_of_coins: {total_sum}"
        (coins_text_w, coins_text_h), _ = cv2.getTextSize(coins_text, font, 0.7, 2)
        (sum_text_w, sum_text_h), _ = cv2.getTextSize(sum_text, font, 0.7, 2)

        # Create a background for the text
        bg_w = max(coins_text_w, sum_text_w) + 20  # Add some padding
        bg_h = coins_text_h + sum_text_h + 30  # Add more padding
        bg = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        bg[:] = bg_color

        # Draw the text on the background
        cv2.putText(bg, coins_text, (10, 20), font, 0.7, (255, 255, 255), 2)
        cv2.putText(bg, sum_text, (10, 50), font, 0.7, (255, 255, 255), 2)

        # Overlay the background with text on the frame
        frame[10:10 + bg_h, 10:10 + bg_w] = bg

        # Display the output frame
        cv2.imshow('Output', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()