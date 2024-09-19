from flask import Flask, request, jsonify
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from azure.storage.blob import BlobServiceClient

app = Flask(__name__)

# Load your ML model
model = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')


# Azure Blob Storage configuration
AZURE_CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=ieeefiles;AccountKey=lYIbbu8YMI2ld08xWPoYHrnMaYObDN1j3R5JpLdrZvFteuQxg1OUZRz3kvppmIGAABGvtNNKyRQb+ASti8vj/w==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_name = 'ieeegg'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    # Process with ML model
    predictions = model.predict(df)

    # Create a plot or image
    plt.figure()
    plt.plot(predictions)  # Customize as needed
    output_image_path = 'output_image.png'
    plt.savefig(output_image_path)
    plt.close()

    # Upload the image to Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob='output_image.png')
    with open(output_image_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    # Construct the image URL
    image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/output_image.png"
    
    return jsonify({'imageUrl': image_url})

if __name__ == '__main__':
    app.run(debug=True)
