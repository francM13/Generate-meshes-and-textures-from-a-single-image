import os
from socket import SocketIO
from flask import Flask, request, jsonify
from getFlame import getFlame
import torch

app = Flask(__name__)

@app.route('/')
def docker_endpoint():
    return 'Hello, I\'m listening!'

@app.route('/getFlame')
def get_flame_endpoint():
    """
    Endpoint for retrieving FLAME model output.
    """
    # Retrieve input parameters from the request
    params = request.get_json()
    shape_params = params.get('shape_params') or torch.rand(1,100,dtype=torch.float32)
    pose_params = params.get('pose_params') or torch.tensor([[1, 0, 1, torch.rand(1),0,0]], dtype=torch.float32)
    expression_params = params.get('expression_params') or torch.rand(1, 50, dtype=torch.float32)
    batch_size = params.get('batch_size') or 1

    # Call the `getFlame` function with the received parameters
    output = getFlame(shape_params, pose_params, expression_params, batch_size)

    # Send the response back to the client
    return jsonify(output[0].tolist(),output[1].tolist(),output[2].tolist())

if __name__ == '__main__':
    app.run(debug=True,port=int(8080),host="0.0.0.0")