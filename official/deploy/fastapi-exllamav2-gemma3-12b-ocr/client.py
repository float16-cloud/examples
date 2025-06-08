import base64
import requests

def send_image_to_endpoint(image_path):
    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare the payload
    payload = {
        "base64_image": encoded_image
    }
    endpoint_url = "https://api.float16.cloud/task/run/function/xxxxxxx"
    headers = {
        "Authorization" : "Bearer float16-r-xxxxxx"
    }
    
    # Send the POST request to the endpoint
    retry_count = 0
    while retry_count < 3:
        try : 
            retry_count += 1
            response = requests.post(endpoint_url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            print(f"Attempt : {retry_count}, Image sent successfully!")
            response_data = response.json()
            message = response_data.get("message", "No message in response")
            model_response = message.split("<start_of_turn>")[-1] # Extract the model response
            print("Model response:", model_response)
        except requests.exceptions.RequestException as e:
            print("Error sending image:", e)
            if retry_count >= 3:
                print("Failed to send image after 3 attempts.")
                return

# Example usage
image_path = "./test_image.jpg"  # Replace with the path to your local image

send_image_to_endpoint(image_path)