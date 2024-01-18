import  time
import  numpy as np
import  sounddevice as sd
import  scipy.io.wavfile as wav
from    interactive_lip_tensorrt import InteractiveLip

# 录制参数
sample_rate         = 44100 # 采样率
channels            = 1     # 声道数
threshold           = 0.6  # 声音强度阈值
min_silence_duration= 1.6   # 最小静音持续时间（秒）
is_recording        = False
silence_counter     = 0
recorded_frames     = []

# 初始化本地模型
audio_filename = './recorded_audio_standard.wav'

# 录制音频回调函数
def audio_callback(indata, frames, input_buffer_adc_time, status):
    global is_recording, silence_counter, recorded_frames

    # 计算声音强度
    rms = np.sqrt(np.mean(indata ** 2))

    if not is_recording:
        # 如果声音强度大于阈值，开始录制
        if rms > threshold:
            print("开始录制音频...")
            is_recording = True
            silence_counter = 0
            recorded_frames = []
    else:
        # 如果声音强度小于阈值，计数器加1
        if rms < threshold:
            silence_counter += 1
        else:
            silence_counter = 0

        # 如果静音持续时间超过阈值，停止录制
        if silence_counter >= int(min_silence_duration * sample_rate / frames):
            print("收音完成")
            is_recording = False
            # 将音频数据保存到文件
            wav.write(audio_filename, sample_rate, np.concatenate(recorded_frames))
            raise sd.CallbackStop
            # sd.stop()
        else:
            # 累积录音数据
            recorded_frames.append(indata.copy())

def lisenting():
    print("等待触发录制...")
    with sd.InputStream(callback=audio_callback, 
                        channels=channels, 
                            samplerate=sample_rate) as stream:
        while (stream.active):
            time.sleep(1./24)
    stream.stop()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

if __name__ == "__main__":

    lip             = InteractiveLip()
    lip.load_model("./checkpoints/wav2lip.pth")
    # lip.make_resources("./WeChat_20240118113635.mp4")
    lip.make_resources("./inputs/ts-tim.mp4")
    # lip.make_resources("test.mp4")
    lip.tensorrt_init("./GFPGANv1.4.onnx")
    lip.invoke_gui_process()
    while True:
        print('请对着麦克风发声....')
        lisenting()
        print("Start to process new request!")
        result_video    = lip.generate_audio(audio_filename)
        lip()
        