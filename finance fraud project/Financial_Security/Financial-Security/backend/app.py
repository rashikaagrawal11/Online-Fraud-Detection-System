from flask import Flask, request, render_template, redirect
from deepfake_audio_detection import predict_deepfake as predict_audio
from deepfake_video_detection import predict_deepfake_video as predict_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER_AUDIO'] = './static/uploaded_audio'
app.config['UPLOAD_FOLDER_VIDEO'] = './static/uploaded_videos'

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' in request.files:
        file = request.files['audio']
        file_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], file.filename)
        file.save(file_path)
        result = predict_audio(file_path)
        return render_template('result.html', result=result)

    elif 'video' in request.files:
        file = request.files['video']
        file_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], file.filename)
        file.save(file_path)
        result = predict_video(file_path)
        return render_template('result.html', result=result)

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
