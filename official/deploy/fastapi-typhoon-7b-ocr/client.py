import base64
import json
import requests

def send_pdf_to_endpoint(pdf_path):
    # Read the image file
    with open(pdf_path, "rb") as image_file:
        # Encode the image as base64
        encoded_pdf = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare the payload
    payload = {
        "base64_pdf": encoded_pdf
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
            break
        except : 
            print("Error sending image:", response.text)
            if retry_count >= 3:
                print("Failed to send image after 3 attempts.")
                return
            
    data = json.loads(response_data)

    try : 
        print(data['natural_text'])
    except :
        print("Error parsing response:", response.text)

# Example usage
pdf_path = "./law.pdf"  # Replace with the path to your local pdf

send_pdf_to_endpoint(pdf_path)