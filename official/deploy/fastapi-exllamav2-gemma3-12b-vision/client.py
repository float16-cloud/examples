import base64
import requests

def send_image_to_endpoint(image_path):
    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    encoded_image = "data:image/jpeg;base64," + encoded_image  # Add the data URI scheme prefix
    print(encoded_image[:50])  # Print the first 50 characters of the encoded image for debugging
    # Prepare the payload
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": encoded_image
                    }
                ]
            }
        ]
    }
    
    endpoint_url = "https://api.float16.cloud/task/run/function/xxxxxxxxx"
    headers = {
        "Authorization" : "Bearer float16-r-xxxxxxxxxxx"
    }
    
    # Send the POST request to the endpoint
    try:
        response = requests.post(endpoint_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Image sent successfully!")
        response_data = response.json()
        message = response_data.get("message", "No message in response")
        model_response = message.split("<start_of_turn>")[-1]
        print("Model response:", model_response)
    except requests.exceptions.RequestException as e:
        print("Error sending image:", e)

# Example usage
image_path = "./test_image.jpg"  # Replace with the path to your local image

send_image_to_endpoint(image_path)