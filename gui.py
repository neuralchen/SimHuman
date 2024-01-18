import  time
import  numpy as np
import  sounddevice as sd
import  scipy.io.wavfile as wav
from    interactive_lip_tensorrt import InteractiveLip

from    torch.backends import cudnn
import  argparse

def getParameters():
    
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('-v', '--video', type=str, default='./', 
                                            help="example video path")


sample_rate         = 44100 
channels            = 1     
threshold           = 0.6  
min_silence_duration= 1.6 
is_recording        = False
silence_counter     = 0
recorded_frames     = []


audio_filename = './recorded_audio_standard.wav'

def audio_callback(indata, frames, input_buffer_adc_time, status):
    global is_recording, silence_counter, recorded_frames

    rms = np.sqrt(np.mean(indata ** 2))

    if not is_recording:
        if rms > threshold:
            print("Start to record audio...")
            is_recording = True
            silence_counter = 0
            recorded_frames = []
    else:
        if rms < threshold:
            silence_counter += 1
        else:
            silence_counter = 0

        if silence_counter >= int(min_silence_duration * sample_rate / frames):
            print("Record finished!")
            is_recording = False
            wav.write(audio_filename, sample_rate, np.concatenate(recorded_frames))
            raise sd.CallbackStop
            # sd.stop()
        else:
            recorded_frames.append(indata.copy())

def lisenting():
    print("==================================================\n")
    print("==Waiting for vioce............................===\n")
    print("==================================================\n")
    with sd.InputStream(callback=audio_callback, 
                        channels=channels, 
                            samplerate=sample_rate) as stream:
        while (stream.active):
            time.sleep(1./24)
    stream.stop()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

if __name__ == "__main__":
    config = getParameters()
    # speed up the program
    cudnn.benchmark = True

    from logo_class import logo_class
    logo_class.print_group_logo()

    lip             = InteractiveLip()
    lip.load_model("./checkpoints/wav2lip.pth")
    lip.make_resources(config.video)
    lip.tensorrt_init("./restoration/GFPGANv1.4.onnx")
    lip.invoke_gui_process()
    while True:
        lisenting()
        print("Start to process new request!")
        result_video    = lip.generate_audio(audio_filename)
        lip()
        